# Method

This project separates intrinsic surface chemistry from reactor transport.

ACE+ should own:

- flow
- pressure
- species transport
- local wafer temperature
- local wall composition/coverage if available

`my_abinitio` should own:

- atomistic pathways
- energy barriers
- temperature-dependent rate constants
- ACE+ surface-rate constants

## Production Stack

```text
1. Define C(S) -> C_sub and C(S) -> C_int endpoint geometries.
2. Run MACE/MLIP NEB to screen pathways quickly.
3. Correct selected MACE images with DFT single-point or full DFT NEB.
4. Convert corrected barriers to harmonic-TST rates over 500-650 C.
5. Fit ACE+ modified Arrhenius parameters.
6. Export C constants for ACE+.
7. Optionally run kMC to test Csub/Cint branching under burial competition.
```

## Rate Law

For each elementary carbon pathway:

```text
k(T) = A * T^n * exp(-(E/R)/T)
```

For a TST/incorporation-class event:

```text
k(T) = (k_B T / h) * exp(-E_a / (k_B T))
```

So an ideal harmonic-TST fit should produce approximately:

```text
n = 1
A = k_B / h
E/R = E_a / k_B
```

If vibrational entropy is included:

```text
k(T) = (k_B T / h) * exp(DeltaS^dagger/k_B) * exp(-DeltaH^dagger/(k_B T))
```

The entropy term changes the effective prefactor.

## What Is Universal

The intrinsic elementary rate for a clearly defined local surface environment can
be portable:

```text
C(S) -> C_sub
C(S) -> C_int
```

## What Is Not Universal

The following are reactor/process-specific and should come from ACE+:

- local precursor partial pressure
- local growth rate
- H coverage
- Ge coverage
- radial temperature variation
- boundary-layer transport

If environment dependence is important, fit barrier corrections:

```text
E_eff = E0 + alpha*N_Ge + beta*strain + gamma*theta_H
```

Only add this after the base temperature-dependent model works.
