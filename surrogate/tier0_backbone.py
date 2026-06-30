"""Tier 0 two-resistance surrogate for low-temperature Si epitaxy.

The model is intentionally small and dependency-free.  It represents the first
physics backbone before any Tier 1 correction is learned from CFD-ACE+ data:

    gas transport to wafer + local empirical/kinetic surface rate

For the Jackel et al. DCS example, the default surface law uses the empirical
Si model reported in JVST A 42, 022702 (2024):

    GR = K * exp(-(Ea/R)/T) * (p_HCl / p_DCS)**a_HCl

where GR is a growth velocity in m/s.  The effective wafer rate is then limited
by the gas-side DCS supply through a simple two-resistance combination.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


R_UNIVERSAL = 8.314462618  # J/(mol K)
P_ATM = 101325.0
T_STP = 273.15
TORR_TO_PA = 133.32236842105263

M_H2 = 2.01588e-3  # kg/mol
M_DCS = 101.00738e-3  # kg/mol, SiH2Cl2
M_HCL = 36.46094e-3  # kg/mol
M_SI = 28.0855e-3  # kg/mol
RHO_SI = 2329.0  # kg/m^3

# Paper table V uses Gamma = 1.126e-8 kmol/m^2.
SITE_DENSITY_KMOL_M2 = 1.126e-8


@dataclass(frozen=True)
class ReactorGeometry:
    """Geometry and transport scale for the simplified half-domain."""

    domain_length_m: float = 0.25
    chamber_height_m: float = 0.20
    wafer_radius_m: float = 0.150
    inlet_velocity_m_s: float = 0.1145
    characteristic_gap_m: float = 0.20
    mass_transfer_edge_enhancement: float = 0.18
    mass_transfer_edge_power: float = 2.0


@dataclass(frozen=True)
class ProcessConditions:
    """Operating point for the Jackel DCS Si epitaxy example."""

    wafer_temperature_C: float = 650.0
    pressure_Torr: float = 300.0
    main_h2_sccm: float = 9000.0
    main_dcs_sccm: float = 400.0
    bottom_h2_sccm: float = 5000.0
    side_h2_sccm: float = 400.0
    side_dcs_sccm: float = 100.0
    hcl_sccm: float = 0.0
    ph3_sccm: float = 0.0
    dcs_surface_center_factor: float = 0.985
    dcs_surface_edge_factor: float = 1.0
    hcl_to_dcs_ratio: float = 0.01
    hcl_to_dcs_edge_multiplier: float = 0.98


@dataclass(frozen=True)
class EmpiricalSiParams:
    """Jackel et al. empirical Si model parameters."""

    K_m_s: float = 1.2e25
    E_over_R_K: float = 95559.0
    a_hcl: float = -3.419
    min_hcl_to_dcs: float = 1.0e-6
    max_hcl_to_dcs: float = 10.0


@dataclass(frozen=True)
class CalibrationAnchor:
    """Optional one-point calibration from a CFD-ACE+ deposition result."""

    target_si_mass_flux_kg_m2_s: float | None = None
    radius_fraction: float = 1.0
    mode: str = "infer_hcl_to_dcs"
    note: str = ""


@dataclass(frozen=True)
class Tier0Point:
    r_m: float
    r_over_R: float
    p_dcs_Pa: float
    p_hcl_Pa: float
    d_dcs_h2_m2_s: float
    k_m_m_s: float
    reaction_si_mass_flux_kg_m2_s: float
    transport_limit_si_mass_flux_kg_m2_s: float
    effective_si_mass_flux_kg_m2_s: float
    growth_nm_min: float
    damkohler: float
    regime: str


@dataclass(frozen=True)
class Tier0Result:
    geometry: ReactorGeometry
    process: ProcessConditions
    params: EmpiricalSiParams
    calibration: CalibrationAnchor
    profile: list[Tier0Point]
    summary: dict[str, float | str | None]


def celsius_to_kelvin(value_C: float) -> float:
    return value_C + 273.15


def torr_to_pa(value_Torr: float) -> float:
    return value_Torr * TORR_TO_PA


def sccm_to_mol_s(value_sccm: float) -> float:
    return value_sccm * 1.0e-6 * P_ATM / (R_UNIVERSAL * T_STP) / 60.0


def growth_m_s_to_nm_min(value_m_s: float) -> float:
    return value_m_s * 1.0e9 * 60.0


def si_mass_flux_to_growth_m_s(value_kg_m2_s: float) -> float:
    return abs(value_kg_m2_s) / RHO_SI


def growth_m_s_to_si_mass_flux(value_m_s: float) -> float:
    return value_m_s * RHO_SI


def gas_density_h2(T_K: float, pressure_Pa: float) -> float:
    return pressure_Pa * M_H2 / (R_UNIVERSAL * T_K)


def viscosity_h2(T_K: float) -> float:
    """Sutherland estimate for hydrogen viscosity."""

    mu_ref = 8.76e-6
    T_ref = 300.0
    sutherland = 72.0
    return mu_ref * (T_K / T_ref) ** 1.5 * (T_ref + sutherland) / (T_K + sutherland)


def diffusivity_dcs_in_h2(T_K: float, pressure_Pa: float) -> float:
    """Approximate DCS-in-H2 binary diffusivity.

    The default is a Chapman-Enskog/Fuller-style scaling anchored to a typical
    light-gas binary diffusivity at 300 K and 1 atm.  Tier 1 should replace or
    tune this with CFD-ACE+ extracted near-wafer mass-transfer data.
    """

    d_ref = 6.0e-5
    return d_ref * (T_K / 300.0) ** 1.75 * (P_ATM / pressure_Pa)


def total_sccm(process: ProcessConditions) -> float:
    return (
        process.main_h2_sccm
        + process.main_dcs_sccm
        + process.bottom_h2_sccm
        + process.side_h2_sccm
        + process.side_dcs_sccm
        + process.hcl_sccm
        + process.ph3_sccm
    )


def inlet_mole_fractions(process: ProcessConditions) -> dict[str, float]:
    total = total_sccm(process)
    if total <= 0.0:
        raise ValueError("total inlet flow must be positive")
    return {
        "H2": (process.main_h2_sccm + process.bottom_h2_sccm + process.side_h2_sccm) / total,
        "DCS": (process.main_dcs_sccm + process.side_dcs_sccm) / total,
        "HCl": process.hcl_sccm / total,
        "PH3": process.ph3_sccm / total,
    }


def local_partial_pressures(
    process: ProcessConditions,
    geometry: ReactorGeometry,
    r_m: float,
) -> tuple[float, float]:
    pressure = torr_to_pa(process.pressure_Torr)
    fractions = inlet_mole_fractions(process)
    r_over_R = min(max(r_m / geometry.wafer_radius_m, 0.0), 1.0)

    center = process.dcs_surface_center_factor
    edge = process.dcs_surface_edge_factor
    dcs_factor = center + (edge - center) * r_over_R**2
    p_dcs = pressure * fractions["DCS"] * dcs_factor

    hcl_ratio = process.hcl_to_dcs_ratio * (
        1.0 + (process.hcl_to_dcs_edge_multiplier - 1.0) * r_over_R**2
    )
    hcl_ratio = max(hcl_ratio, 0.0)
    if fractions["HCl"] > 0.0:
        p_hcl = pressure * fractions["HCl"]
    else:
        p_hcl = p_dcs * hcl_ratio
    return p_dcs, p_hcl


def empirical_si_growth_m_s(
    T_K: float,
    p_dcs_Pa: float,
    p_hcl_Pa: float,
    params: EmpiricalSiParams = EmpiricalSiParams(),
) -> float:
    if p_dcs_Pa <= 0.0:
        return 0.0
    ratio = p_hcl_Pa / p_dcs_Pa
    ratio = min(max(ratio, params.min_hcl_to_dcs), params.max_hcl_to_dcs)
    return params.K_m_s * math.exp(-params.E_over_R_K / T_K) * ratio**params.a_hcl


def mass_transfer_coefficient(
    T_K: float,
    pressure_Pa: float,
    geometry: ReactorGeometry,
    r_over_R: float,
) -> tuple[float, float]:
    D = diffusivity_dcs_in_h2(T_K, pressure_Pa)
    strain_rate = max(geometry.inlet_velocity_m_s / geometry.characteristic_gap_m, 1.0e-9)
    boundary_layer = math.sqrt(math.pi * D / (2.0 * strain_rate))
    base_km = D / max(boundary_layer, 1.0e-12)
    edge_factor = 1.0 + geometry.mass_transfer_edge_enhancement * r_over_R**geometry.mass_transfer_edge_power
    return base_km * edge_factor, D


def effective_two_resistance_flux(
    reaction_si_mol_m2_s: float,
    transport_dcs_mol_m2_s: float,
) -> tuple[float, float]:
    rxn = max(reaction_si_mol_m2_s, 0.0)
    trans = max(transport_dcs_mol_m2_s, 0.0)
    if rxn == 0.0 or trans == 0.0:
        return 0.0, math.inf if trans == 0.0 and rxn > 0.0 else 0.0
    effective = (rxn * trans) / (rxn + trans)
    return effective, rxn / trans


def classify_regime(damkohler: float) -> str:
    if not math.isfinite(damkohler):
        return "transport-unavailable"
    if damkohler < 0.1:
        return "surface-limited"
    if damkohler > 10.0:
        return "transport-limited"
    return "mixed"


def evaluate_profile(
    process: ProcessConditions = ProcessConditions(),
    geometry: ReactorGeometry = ReactorGeometry(),
    params: EmpiricalSiParams = EmpiricalSiParams(),
    calibration: CalibrationAnchor = CalibrationAnchor(),
    n_points: int = 61,
) -> Tier0Result:
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    if calibration.target_si_mass_flux_kg_m2_s is not None:
        process = apply_calibration(process, geometry, params, calibration)

    T_K = celsius_to_kelvin(process.wafer_temperature_C)
    pressure = torr_to_pa(process.pressure_Torr)
    points: list[Tier0Point] = []
    for i in range(n_points):
        r_over_R = i / (n_points - 1)
        r_m = r_over_R * geometry.wafer_radius_m
        p_dcs, p_hcl = local_partial_pressures(process, geometry, r_m)
        growth_rxn = empirical_si_growth_m_s(T_K, p_dcs, p_hcl, params)
        reaction_si_mol = growth_rxn * RHO_SI / M_SI
        k_m, D = mass_transfer_coefficient(T_K, pressure, geometry, r_over_R)
        c_dcs = p_dcs / (R_UNIVERSAL * T_K)
        transport_dcs_mol = k_m * c_dcs
        effective_si_mol, da = effective_two_resistance_flux(reaction_si_mol, transport_dcs_mol)
        effective_flux = effective_si_mol * M_SI
        points.append(
            Tier0Point(
                r_m=r_m,
                r_over_R=r_over_R,
                p_dcs_Pa=p_dcs,
                p_hcl_Pa=p_hcl,
                d_dcs_h2_m2_s=D,
                k_m_m_s=k_m,
                reaction_si_mass_flux_kg_m2_s=reaction_si_mol * M_SI,
                transport_limit_si_mass_flux_kg_m2_s=transport_dcs_mol * M_SI,
                effective_si_mass_flux_kg_m2_s=effective_flux,
                growth_nm_min=growth_m_s_to_nm_min(effective_flux / RHO_SI),
                damkohler=da,
                regime=classify_regime(da),
            )
        )

    summary = summarize_profile(points)
    summary.update(
        {
            "calibrated_hcl_to_dcs_ratio": process.hcl_to_dcs_ratio,
            "temperature_C": process.wafer_temperature_C,
            "pressure_Torr": process.pressure_Torr,
            "wafer_radius_m": geometry.wafer_radius_m,
        }
    )
    return Tier0Result(geometry, process, params, calibration, points, summary)


def apply_calibration(
    process: ProcessConditions,
    geometry: ReactorGeometry,
    params: EmpiricalSiParams,
    calibration: CalibrationAnchor,
) -> ProcessConditions:
    if calibration.mode != "infer_hcl_to_dcs":
        raise ValueError(f"unsupported calibration mode: {calibration.mode}")
    target = calibration.target_si_mass_flux_kg_m2_s
    if target is None:
        return process

    T_K = celsius_to_kelvin(process.wafer_temperature_C)
    r_m = min(max(calibration.radius_fraction, 0.0), 1.0) * geometry.wafer_radius_m
    p_dcs, _ = local_partial_pressures(process, geometry, r_m)
    target_growth = si_mass_flux_to_growth_m_s(target)
    base = params.K_m_s * math.exp(-params.E_over_R_K / T_K)
    if target_growth <= 0.0 or base <= 0.0 or params.a_hcl == 0.0:
        return process
    ratio = (target_growth / base) ** (1.0 / params.a_hcl)
    ratio = min(max(ratio, params.min_hcl_to_dcs), params.max_hcl_to_dcs)

    # Store the ratio relative to the center value; local_partial_pressures will
    # apply the configured edge multiplier again at the calibration radius.
    r_over_R = min(max(r_m / geometry.wafer_radius_m, 0.0), 1.0)
    edge_factor = 1.0 + (process.hcl_to_dcs_edge_multiplier - 1.0) * r_over_R**2
    center_ratio = ratio / max(edge_factor, 1.0e-12)
    return replace(process, hcl_to_dcs_ratio=center_ratio)


def summarize_profile(points: list[Tier0Point]) -> dict[str, float | str | None]:
    if not points:
        return {}
    growth = [p.growth_nm_min for p in points]
    flux = [p.effective_si_mass_flux_kg_m2_s for p in points]
    da_values = [p.damkohler for p in points if math.isfinite(p.damkohler)]
    center = points[0]
    edge = points[-1]
    mean_growth = sum(growth) / len(growth)
    delta = edge.growth_nm_min - center.growth_nm_min
    return {
        "center_growth_nm_min": center.growth_nm_min,
        "edge_growth_nm_min": edge.growth_nm_min,
        "mean_growth_nm_min": mean_growth,
        "min_growth_nm_min": min(growth),
        "max_growth_nm_min": max(growth),
        "edge_minus_center_growth_nm_min": delta,
        "growth_nonuniformity_percent": 100.0 * (max(growth) - min(growth)) / max(mean_growth, 1.0e-30),
        "center_si_mass_flux_kg_m2_s": center.effective_si_mass_flux_kg_m2_s,
        "edge_si_mass_flux_kg_m2_s": edge.effective_si_mass_flux_kg_m2_s,
        "max_si_mass_flux_kg_m2_s": max(flux),
        "min_damkohler": min(da_values) if da_values else None,
        "max_damkohler": max(da_values) if da_values else None,
        "dominant_regime": max({p.regime for p in points}, key=[p.regime for p in points].count),
    }


def write_profile_csv(path: str | Path, points: Iterable[Tier0Point]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in points]
    if not rows:
        p.write_text("")
        return
    with p.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_result_json(path: str | Path, result: Tier0Result) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "geometry": asdict(result.geometry),
        "process": asdict(result.process),
        "params": asdict(result.params),
        "calibration": asdict(result.calibration),
        "summary": result.summary,
    }
    p.write_text(json.dumps(payload, indent=2))


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def model_from_config(config: dict[str, Any]) -> tuple[ProcessConditions, ReactorGeometry, EmpiricalSiParams, CalibrationAnchor, int]:
    process = ProcessConditions(**config.get("process", {}))
    geometry = ReactorGeometry(**config.get("geometry", {}))
    params = EmpiricalSiParams(**config.get("empirical_si_params", {}))
    calibration = CalibrationAnchor(**config.get("calibration", {}))
    n_points = int(config.get("n_points", 61))
    return process, geometry, params, calibration, n_points


def evaluate_config(path: str | Path) -> Tier0Result:
    process, geometry, params, calibration, n_points = model_from_config(load_config(path))
    return evaluate_profile(process, geometry, params, calibration, n_points)


# Compatibility helpers for the earlier notebook-style prototype.
PARAMS = asdict(EmpiricalSiParams())


def growth_rate(th: dict[str, float]) -> float:
    process = ProcessConditions(
        wafer_temperature_C=float(th.get("T_s", 923.15)) - 273.15,
        pressure_Torr=float(th.get("p_tot", torr_to_pa(300.0))) / TORR_TO_PA,
        hcl_to_dcs_ratio=float(th.get("hcl_to_dcs_ratio", 0.01)),
    )
    result = evaluate_profile(process=process, n_points=2)
    return result.profile[0].effective_si_mass_flux_kg_m2_s / RHO_SI


def diagnostics(th: dict[str, float]) -> dict[str, float | str | None]:
    process = ProcessConditions(
        wafer_temperature_C=float(th.get("T_s", 923.15)) - 273.15,
        pressure_Torr=float(th.get("p_tot", torr_to_pa(300.0))) / TORR_TO_PA,
        hcl_to_dcs_ratio=float(th.get("hcl_to_dcs_ratio", 0.01)),
    )
    result = evaluate_profile(process=process, n_points=2)
    return result.summary
