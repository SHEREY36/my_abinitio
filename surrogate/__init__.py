"""Surrogate utilities for the my_abinitio project.

Tier 0 is dependency-light.  Tier 1/Tier 2 need the ``surrogate`` optional
dependencies from ``pyproject.toml``; those modules are loaded lazily so the
Tier 0 command can still run in a bare Python environment.
"""

from importlib import import_module


_EXPORTS = {
    "GP": ("core", "GP"),
    "MultiGP": ("core", "MultiGP"),
    "POD": ("core", "POD"),
    "Param": ("paramspace", "Param"),
    "ParameterSpace": ("paramspace", "ParameterSpace"),
    "Proposal": ("active", "Proposal"),
    "Snapshot": ("snapshots", "Snapshot"),
    "SnapshotDataset": ("snapshots", "SnapshotDataset"),
    "Tier1Surrogate": ("tier1", "Tier1Surrogate"),
    "Tier2FieldSurrogate": ("tier2", "Tier2FieldSurrogate"),
    "active_learning_step": ("active", "active_learning_step"),
    "growth_profile_from_deposition": ("snapshots", "growth_profile_from_deposition"),
    "inlet_bc": ("runner", "inlet_bc"),
    "jackel_si_space": ("paramspace", "jackel_si_space"),
    "load_csv": ("snapshots", "load_csv"),
    "load_dtf": ("snapshots", "load_dtf"),
    "load_tecplot_ascii": ("snapshots", "load_tecplot_ascii"),
    "make_tier0_prior": ("tier1", "make_tier0_prior"),
    "propose_batch": ("active", "propose_batch"),
    "qois_from_profile": ("snapshots", "qois_from_profile"),
    "write_run_table": ("runner", "write_run_table"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ImportError as exc:
        raise ImportError(
            f"surrogate.{name} requires the optional surrogate dependencies. "
            'Install them with: python -m pip install ".[surrogate]"'
        ) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
