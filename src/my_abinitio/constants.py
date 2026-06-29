"""Physical constants and unit helpers."""

KB_EV_PER_K = 8.617333262145e-5
KB_J_PER_K = 1.380649e-23
H_EV_S = 4.135667696e-15
KB_OVER_H_PER_K_S = KB_EV_PER_K / H_EV_S


def ev_to_kelvin(energy_ev: float) -> float:
    """Convert an activation energy in eV to ACE+ E/R units in K."""
    return energy_ev / KB_EV_PER_K


def kelvin_to_ev(e_over_r_k: float) -> float:
    """Convert ACE+ E/R units in K to eV."""
    return e_over_r_k * KB_EV_PER_K
