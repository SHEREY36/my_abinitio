"""Build Si/SiGeC surface structures for C-sub/C-int NEB pathways.

The structures produced here are starting hypotheses, not final chemistry.
They encode a minimal RP-CVD growth-front picture:

* a Si(001)-like slab represents the exposed growth surface,
* selected top-layer Si atoms are replaced by Ge to mimic SiGe alloying,
* optional H/Cl atoms represent passivating adsorbates from hydride/chloride
  chemistry,
* one carbon atom starts as a surface species and ends either substitutional
  or interstitial.

All pathway endpoints have the same atom count and atom identities so they can
be used directly by ASE NEB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable


@dataclass(frozen=True)
class SlabBuildConfig:
    """Input parameters controlling the SiGeC slab hypotheses."""

    repetitions: tuple[int, int, int] = (4, 4, 5)
    lattice_constant_A: float = 5.431
    vacuum_A: float = 14.0
    ge_fraction: float = 0.10
    ge_layers: int = 2
    termination: str = "H"
    termination_fraction: float = 0.25
    seed: int = 7
    c_sub_layer_from_top: int = 2
    c_height_A: float = 1.75
    adatom_height_A: float = 1.85
    interstitial_depth_A: float = 1.35


def require_ase():
    """Import ASE lazily so the package still supports no-dependency toy tests."""

    try:
        from ase import Atom, Atoms
        from ase.build import bulk, make_supercell, surface
        from ase.constraints import FixAtoms
        from ase.io import write
    except Exception as exc:  # pragma: no cover - exercised only without ASE.
        raise SystemExit(
            "ASE is required for slab construction. Install with:\n"
            "  python -m pip install -e '.[atomistics]'"
        ) from exc
    return Atom, Atoms, bulk, make_supercell, surface, FixAtoms, write


def _top_layer_indices(atoms, z_tol: float = 0.35) -> list[int]:
    zmax = max(atoms.positions[:, 2])
    return [i for i, pos in enumerate(atoms.positions) if zmax - pos[2] <= z_tol]


def _bottom_layer_indices(atoms, z_tol: float = 0.80) -> list[int]:
    zmin = min(atoms.positions[:, 2])
    return [i for i, pos in enumerate(atoms.positions) if pos[2] - zmin <= z_tol]


def _layer_indices_from_top(atoms, n_layers: int, z_tol: float = 0.35) -> list[int]:
    rounded = sorted({round(float(z), 3) for z in atoms.positions[:, 2]}, reverse=True)
    selected_z = rounded[: max(1, n_layers)]
    return [
        i
        for i, pos in enumerate(atoms.positions)
        if any(abs(float(pos[2]) - z) <= z_tol for z in selected_z)
    ]


def _single_layer_indices_from_top(atoms, layer_from_top: int, z_tol: float = 0.35) -> list[int]:
    if layer_from_top < 1:
        raise ValueError("layer_from_top must be 1 for top layer, 2 for subsurface, etc.")
    rounded = sorted({round(float(z), 3) for z in atoms.positions[:, 2]}, reverse=True)
    if layer_from_top > len(rounded):
        raise ValueError(f"requested layer {layer_from_top}, but slab only has {len(rounded)} z layers")
    target_z = rounded[layer_from_top - 1]
    return [i for i, pos in enumerate(atoms.positions) if abs(float(pos[2]) - target_z) <= z_tol]


def _nearest_xy_index(atoms, indices: Iterable[int], xy: tuple[float, float]) -> int:
    x, y = xy
    return min(
        indices,
        key=lambda i: (float(atoms.positions[i, 0]) - x) ** 2
        + (float(atoms.positions[i, 1]) - y) ** 2,
    )


def _cell_center_xy(atoms) -> tuple[float, float]:
    cell = atoms.cell.array
    return (0.5 * float(cell[0, 0] + cell[1, 0]), 0.5 * float(cell[0, 1] + cell[1, 1]))


def _replace_si_with_ge(atoms, cfg: SlabBuildConfig, protected: set[int]) -> list[int]:
    rng = Random(cfg.seed)
    candidates = [
        i
        for i in _layer_indices_from_top(atoms, cfg.ge_layers)
        if atoms[i].symbol == "Si" and i not in protected
    ]
    n_ge = int(round(cfg.ge_fraction * len(candidates)))
    if cfg.ge_fraction > 0 and n_ge == 0 and candidates:
        n_ge = 1
    chosen = sorted(rng.sample(candidates, min(n_ge, len(candidates))))
    for i in chosen:
        atoms[i].symbol = "Ge"
    return chosen


def _add_termination(atoms, cfg: SlabBuildConfig, protected_xy: tuple[float, float]) -> list[int]:
    termination = cfg.termination.strip()
    if termination.lower() in {"none", "bare", "clean"} or cfg.termination_fraction <= 0:
        return []
    if termination not in {"H", "Cl"}:
        raise ValueError("termination must be H, Cl, none, bare, or clean")

    Atom, *_rest = require_ase()
    rng = Random(cfg.seed + 11)
    top = _top_layer_indices(atoms)
    protected_top = _nearest_xy_index(atoms, top, protected_xy)
    candidates = [i for i in top if i != protected_top]
    n_term = int(round(cfg.termination_fraction * len(candidates)))
    chosen = sorted(rng.sample(candidates, min(n_term, len(candidates))))
    height = 1.48 if termination == "H" else 2.05
    for i in chosen:
        pos = atoms.positions[i].copy()
        pos[2] += height
        atoms.append(Atom(termination, position=pos))
    return chosen


def build_sige_c_endpoints(cfg: SlabBuildConfig):
    """Return initial, C-sub final, and C-int final ASE Atoms objects."""

    Atom, _Atoms, bulk, _make_supercell, surface, FixAtoms, _write = require_ase()

    si_bulk = bulk("Si", "diamond", a=cfg.lattice_constant_A, cubic=True)
    slab = surface(
        si_bulk,
        (0, 0, 1),
        layers=cfg.repetitions[2],
        vacuum=cfg.vacuum_A,
        periodic=True,
    )
    slab = slab.repeat((cfg.repetitions[0], cfg.repetitions[1], 1))
    slab.wrap()

    center_xy = _cell_center_xy(slab)
    top = _top_layer_indices(slab)
    sub_candidates = _single_layer_indices_from_top(slab, cfg.c_sub_layer_from_top)
    target_site = _nearest_xy_index(slab, sub_candidates, center_xy)
    top_site = _nearest_xy_index(slab, top, tuple(slab.positions[target_site, :2]))
    protected = {target_site, top_site}
    ge_indices = _replace_si_with_ge(slab, cfg, protected)

    # Fix the lower slab so relaxation focuses on the growth front.
    fixed = _bottom_layer_indices(slab)
    slab.set_constraint(FixAtoms(indices=fixed))

    # Common initial state: C surface species above a likely open top site.
    initial = slab.copy()
    _add_termination(initial, cfg, tuple(initial.positions[top_site, :2]))
    target = initial.positions[target_site].copy()
    surface_anchor = initial.positions[top_site].copy()
    c_surface_pos = surface_anchor.copy()
    c_surface_pos[2] += cfg.c_height_A
    initial.append(Atom("C", position=c_surface_pos))
    c_index = len(initial) - 1

    # C_sub final: C moves into the lattice site; displaced Si/Ge becomes a
    # surface adatom. This preserves atom identities for NEB.
    c_sub = initial.copy()
    displaced_pos = c_sub.positions[target_site].copy()
    adatom_pos = displaced_pos.copy()
    adatom_pos[0] += 0.35
    adatom_pos[1] += 0.35
    adatom_pos[2] += cfg.adatom_height_A
    c_sub.positions[c_index] = displaced_pos
    c_sub.positions[target_site] = adatom_pos

    # C_int final: C moves below the top layer toward a simple tetrahedral-like
    # subsurface position. Relaxation/NEB will refine this geometry.
    c_int = initial.copy()
    int_pos = target.copy()
    int_pos[0] += 0.25
    int_pos[1] += 0.25
    int_pos[2] -= cfg.interstitial_depth_A
    c_int.positions[c_index] = int_pos

    metadata = {
        "target_site_index": target_site,
        "top_site_index": top_site,
        "carbon_index": c_index,
        "ge_indices": ge_indices,
        "fixed_indices": fixed,
        "n_atoms_initial": len(initial),
    }
    return initial, c_sub, c_int, metadata


def write_sige_c_endpoints(output_dir: Path, cfg: SlabBuildConfig) -> dict[str, str | int | list[int]]:
    """Build and write `.traj` plus `.xyz` files for visual inspection."""

    _Atom, _Atoms, _bulk, _make_supercell, _surface, _FixAtoms, write = require_ase()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial, c_sub, c_int, metadata = build_sige_c_endpoints(cfg)
    paths = {
        "C_surface_IS": output_dir / "C_surface_IS.traj",
        "C_sub_FS": output_dir / "C_sub_FS.traj",
        "C_int_FS": output_dir / "C_int_FS.traj",
    }
    write(paths["C_surface_IS"], initial)
    write(paths["C_sub_FS"], c_sub)
    write(paths["C_int_FS"], c_int)
    write(output_dir / "C_surface_IS.xyz", initial)
    write(output_dir / "C_sub_FS.xyz", c_sub)
    write(output_dir / "C_int_FS.xyz", c_int)
    return {**metadata, **{k: str(v) for k, v in paths.items()}}
