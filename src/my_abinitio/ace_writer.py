"""Write ACE+/C snippets from modified Arrhenius parameters."""

from __future__ import annotations

from pathlib import Path


def write_ace_surface_rates(path: str | Path, rates: dict) -> None:
    sub = rates["C_sub"]["modified_arrhenius"]
    cint = rates["C_int"]["modified_arrhenius"]
    text = f"""/*
 * Generated ACE+ surface-rate constants for Csub/Cint.
 * Form: k(T) = A * pow(T,n) * exp(-(E/R)/T)
 */
#include <math.h>

static const double A_C_SUB = {sub["A"]:.12e};
static const double N_C_SUB = {sub["n"]:.12e};
static const double ER_C_SUB = {sub["E_over_R_K"]:.12e};

static const double A_C_INT = {cint["A"]:.12e};
static const double N_C_INT = {cint["n"]:.12e};
static const double ER_C_INT = {cint["E_over_R_K"]:.12e};

static double k_c_sub(double T)
{{
    return A_C_SUB * pow(T, N_C_SUB) * exp(-ER_C_SUB / T);
}}

static double k_c_int(double T)
{{
    return A_C_INT * pow(T, N_C_INT) * exp(-ER_C_INT / T);
}}

static double f_c_sub(double T, double k_bury)
{{
    double r_sub = k_c_sub(T) + k_bury;
    double r_int = k_c_int(T);
    return r_sub / (r_sub + r_int);
}}
"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
