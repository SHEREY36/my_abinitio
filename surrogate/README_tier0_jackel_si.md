# Tier 0 Jackel DCS Si Epitaxy Surrogate

This folder now contains a runnable Tier 0 base for the Jackel et al. single-wafer DCS Si epitaxy example.

Run it from the repo root:

```bash
python surrogate/run_tier0_jackel_si.py
```

It writes:

```text
surrogate/outputs/jackel_si_tier0_profile.csv
surrogate/outputs/jackel_si_tier0_summary.json
```

## What Is Implemented

The model combines:

- the attached schematic dimensions: 0.25 m horizontal domain, 0.2 m height, 0.15 m wafer radius;
- Jackel et al. process conditions: about 300 Torr, DCS in H2, wafer temperature below 700 C;
- Jackel et al. empirical Si model:

```text
GR = K * exp(-(Ea/R)/T) * (p_HCl/p_DCS)^a_HCl
K = 1.2e25 m/s
Ea/R = 95559 K
a_HCl = -3.419
```

- a gas-side DCS mass-transfer limit using an H2/DCS diffusivity estimate and the inlet velocity scale visible in your CFD screenshot;
- a two-resistance combination so the output reports whether each radial point is surface-limited, mixed, or transport-limited.

The current config includes a one-point calibration anchor from your screenshot colorbar:

```text
max |Dep_SI(B)| ~= 7.285e-8 kg/(m^2 s)
```

Replace this with exported CFD-ACE+ line-probe or wafer-average data as soon as you have it.

## Minimum CFD-ACE+ Data To Export Next

For each run, export a radial line probe along the wafer/deposition boundary after azimuthal averaging if available. Use SI units.

Recommended CSV columns:

```text
case_id,T_C,pressure_Torr,r_m,Dep_SI_B_kg_m2_s,VelocityMagnitude_m_s,p_DCS_Pa,p_HCl_Pa,p_H2_Pa
```

Minimum usable columns:

```text
case_id,T_C,pressure_Torr,r_m,Dep_SI_B_kg_m2_s
```

The most valuable extra fields are `p_DCS_Pa` and `p_HCl_Pa` at the first gas cell above the wafer, because the Jackel empirical model depends directly on `p_HCl/p_DCS`.

## Suggested Next ACE+ Run Matrix

Use the current run as the base case. Then add:

```text
base_T_minus_10C   same flows, T - 10 C
base_T_plus_10C    same flows, T + 10 C
dcs_minus_20pct    reduce main and side DCS by 20 percent
dcs_plus_20pct     increase main and side DCS by 20 percent
flow_minus_20pct   reduce H2 carrier flows by 20 percent
flow_plus_20pct    increase H2 carrier flows by 20 percent
pressure_minus     250 Torr, same flow recipe
pressure_plus      350 Torr, same flow recipe
```

For each run, save both the deposition flux and the near-wafer gas partial pressures. With those data, Tier 1 can learn the correction between this cheap Tier 0 backbone and CFD-ACE+.

## Important Boundary

This is not trying to reproduce the full seven-reaction kinetic mechanism yet. It is the fast, interpretable base model. Its job is to produce a plausible radial profile, diagnose depletion and Damkohler regime, and tell us which CFD data will most improve the surrogate.
