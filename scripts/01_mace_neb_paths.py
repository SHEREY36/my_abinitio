#!/usr/bin/env python3
"""Run MACE/ASE NEB calculations for configured carbon pathways.

This script requires optional atomistic dependencies:

    python -m pip install -e ".[atomistics]"

It is intentionally a thin real-workflow wrapper.  The toy validation lives in
scripts/00_run_toy_pipeline.py and does not require MACE/ASE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.io_utils import read_json, write_json


def resolve_device(requested: str) -> str:
    """Choose the best available PyTorch device for MACE."""

    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def write_dft_points(out_dir: Path, name: str, images, ts_index: int, write) -> dict[str, str]:
    """Write the MACE-selected IS/TS/FS structures for later DFT correction."""

    dft_dir = out_dir / "dft_points"
    dft_dir.mkdir(parents=True, exist_ok=True)
    points = {
        "IS": images[0],
        "TS": images[ts_index],
        "FS": images[-1],
    }
    paths: dict[str, str] = {}
    for label, atoms in points.items():
        stem = f"{name}_{label}"
        traj_path = dft_dir / f"{stem}.traj"
        xyz_path = dft_dir / f"{stem}.xyz"
        write(traj_path, atoms)
        write(xyz_path, atoms)
        paths[f"{label.lower()}_traj"] = str(traj_path)
        paths[f"{label.lower()}_xyz"] = str(xyz_path)
    return paths


def relax_endpoint(label: str, atoms, calc, out_dir: Path, fmax: float, steps: int, FIRE, write):
    """Relax one endpoint before NEB and write restart-friendly artifacts."""

    relaxed = atoms.copy()
    relaxed.calc = calc
    log_path = out_dir / f"relax_{label}.log"
    traj_path = out_dir / f"relax_{label}.traj"
    print(
        f"  Relaxing {label} endpoint: fmax={fmax}, steps={steps}\n"
        f"  Log: {log_path}\n"
        f"  Trajectory: {traj_path}",
        flush=True,
    )
    opt = FIRE(relaxed, trajectory=str(traj_path), logfile=str(log_path))
    opt.run(fmax=fmax, steps=steps)
    write(out_dir / f"relaxed_{label}.traj", relaxed)
    write(out_dir / f"relaxed_{label}.xyz", relaxed)
    print(f"  {label} endpoint relaxation complete.", flush=True)
    return relaxed


def run_mace_neb(
    config_path: Path,
    output_path: Path,
    *,
    model: str,
    device: str,
    default_dtype: str,
    dispersion: bool,
    fmax: float | None,
    steps: int | None,
    interpolation: str,
    relax_endpoints: bool,
    endpoint_fmax: float,
    endpoint_steps: int,
    only: set[str] | None,
    climb: bool,
    restart_traj: Path | None,
) -> None:
    try:
        from ase.io import read
        from ase.io import write
        from ase.mep import NEB
        from ase.optimize import FIRE
        from mace.calculators import mace_mp
    except Exception as exc:
        raise SystemExit(
            "ASE/MACE is not installed. Run the toy pipeline locally, or install "
            "with: python -m pip install -e '.[atomistics]'"
        ) from exc

    cfg = read_json(config_path)
    selected_device = resolve_device(device)
    print(
        "MACE settings: "
        f"model={model}, device={selected_device}, dtype={default_dtype}, dispersion={dispersion}",
        flush=True,
    )
    try:
        calc = mace_mp(
            model=model,
            dispersion=dispersion,
            default_dtype=default_dtype,
            device=selected_device,
        )
    except Exception as exc:
        if device == "auto" and selected_device != "cpu":
            print(
                f"MACE could not initialize on {selected_device}; falling back to CPU. "
                "Use --device mps or --device cuda if you want a hard failure instead.",
                flush=True,
            )
            selected_device = "cpu"
            calc = mace_mp(
                model=model,
                dispersion=dispersion,
                default_dtype=default_dtype,
                device=selected_device,
            )
        else:
            raise SystemExit(f"MACE failed to initialize on device={selected_device}: {exc}") from exc
    rows = []
    for path_cfg in cfg["pathways"]:
        name = path_cfg["name"]
        if only is not None and name not in only:
            print(f"Skipping NEB pathway: {name}", flush=True)
            continue
        print(f"Running NEB pathway: {name}", flush=True)
        out_dir = output_path.parent / name
        out_dir.mkdir(parents=True, exist_ok=True)
        total_images = int(path_cfg.get("n_images", 9)) + 2
        path_restart = Path(path_cfg.get("restart_traj", restart_traj)) if path_cfg.get("restart_traj", restart_traj) else None
        if path_restart is not None:
            frames = read(path_restart, ":")
            if len(frames) < total_images:
                raise SystemExit(
                    f"Restart trajectory {path_restart} has {len(frames)} frames, "
                    f"but {total_images} are needed for pathway {name}."
                )
            images = frames[-total_images:]
            print(f"  Restarting from final band in {path_restart}", flush=True)
        else:
            is_atoms = read(path_cfg["initial_structure"])
            fs_atoms = read(path_cfg["final_structure"])
            if relax_endpoints:
                is_atoms = relax_endpoint("IS", is_atoms, calc, out_dir, endpoint_fmax, endpoint_steps, FIRE, write)
                fs_atoms = relax_endpoint("FS", fs_atoms, calc, out_dir, endpoint_fmax, endpoint_steps, FIRE, write)
            images = [is_atoms]
            images += [is_atoms.copy() for _ in range(int(path_cfg.get("n_images", 9)))]
            images += [fs_atoms]
        neb = NEB(images, climb=climb, allow_shared_calculator=True, method="improvedtangent")
        if path_restart is None:
            path_interpolation = str(path_cfg.get("interpolation", interpolation))
            print(f"  Interpolating NEB images with {path_interpolation}...", flush=True)
            if path_interpolation == "idpp":
                neb.interpolate("idpp", apply_constraint=False)
            else:
                neb.interpolate(method="linear", apply_constraint=False)
            print("  Interpolation complete.", flush=True)
        for image in images:
            image.calc = calc
        traj_path = out_dir / "neb.traj"
        log_path = out_dir / "neb.log"
        opt = FIRE(neb, trajectory=str(traj_path), logfile=str(log_path))
        path_fmax = float(path_cfg.get("fmax", fmax if fmax is not None else 0.05))
        path_steps = int(path_cfg.get("steps", steps if steps is not None else 1000))
        print(
            f"  Optimizing with FIRE: fmax={path_fmax}, steps={path_steps}\n"
            f"  Log: {log_path}\n"
            f"  Trajectory: {traj_path}",
            flush=True,
        )
        opt.run(fmax=path_fmax, steps=path_steps)
        print("  FIRE optimization complete.", flush=True)
        energies = [img.get_potential_energy() for img in images]
        ts_index = max(range(len(energies)), key=lambda i: energies[i] - energies[0])
        dft_points = write_dft_points(out_dir, name, images, ts_index, write)
        rows.append(
            {
                "name": name,
                "reaction": path_cfg.get("reaction", name),
                "model": model,
                "device": selected_device,
                "default_dtype": default_dtype,
                "fmax": path_fmax,
                "steps": path_steps,
                "relax_endpoints": relax_endpoints,
                "endpoint_fmax": endpoint_fmax if relax_endpoints else None,
                "endpoint_steps": endpoint_steps if relax_endpoints else None,
                "climb": climb,
                "restart_traj": str(path_restart) if path_restart else None,
                "mace_is_ev": energies[0],
                "mace_ts_ev": energies[ts_index],
                "mace_fs_ev": energies[-1],
                "barrier_ev": energies[ts_index] - energies[0],
                "ts_index": ts_index,
                "traj": str(out_dir / "neb.traj"),
                "dft_points": dft_points,
            }
        )
        print(
            f"  barrier={energies[ts_index] - energies[0]:.6f} eV, "
            f"TS image={ts_index}, DFT points={out_dir / 'dft_points'}",
            flush=True,
        )
    write_json(output_path, {"barriers": rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/c_paths.json")
    parser.add_argument("--output", default="results/mace_neb_summary.json")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="auto prefers Apple MPS, then CUDA, then CPU.",
    )
    parser.add_argument("--model", default="medium", help="MACE-MP model size/name.")
    parser.add_argument("--default-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--dispersion", action="store_true")
    parser.add_argument("--fmax", type=float, default=None, help="Global NEB force tolerance in eV/A.")
    parser.add_argument("--steps", type=int, default=None, help="Global maximum optimizer steps.")
    parser.add_argument(
        "--interpolation",
        choices=["idpp", "linear"],
        default="idpp",
        help="idpp gives a better starting band; linear is faster for quick diagnostics.",
    )
    parser.add_argument(
        "--relax-endpoints",
        action="store_true",
        help="Relax IS and FS structures with MACE before building the NEB band.",
    )
    parser.add_argument("--endpoint-fmax", type=float, default=0.08)
    parser.add_argument("--endpoint-steps", type=int, default=300)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only the named pathway. Can be repeated, e.g. --only C_sub --only C_int.",
    )
    parser.add_argument(
        "--climb",
        action="store_true",
        help="Enable climbing-image NEB. Use after a regular non-climbing NEB is stable.",
    )
    parser.add_argument(
        "--restart-traj",
        type=Path,
        default=None,
        help="Restart from the final band in an existing ASE NEB trajectory.",
    )
    args = parser.parse_args()
    run_mace_neb(
        Path(args.config),
        Path(args.output),
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        dispersion=args.dispersion,
        fmax=args.fmax,
        steps=args.steps,
        interpolation=args.interpolation,
        relax_endpoints=args.relax_endpoints,
        endpoint_fmax=args.endpoint_fmax,
        endpoint_steps=args.endpoint_steps,
        only=set(args.only) if args.only else None,
        climb=args.climb,
        restart_traj=args.restart_traj,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
