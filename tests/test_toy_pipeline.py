from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.constants import KB_OVER_H_PER_K_S, ev_to_kelvin
from my_abinitio.dft_correction import corrected_barrier
from my_abinitio.kmc import branching_fraction
from my_abinitio.neb import run_toy_neb
from my_abinitio.toy_potential import DoubleWell2D
from my_abinitio.tst import fit_modified_arrhenius, rates_from_barrier


class ToyWorkflowTests(unittest.TestCase):
    def test_toy_neb_barrier(self):
        result = run_toy_neb("toy", DoubleWell2D(1.2), n_images=9)
        self.assertAlmostEqual(result.barrier_ev, 1.2, places=10)

    def test_dft_correction(self):
        barrier = corrected_barrier(0.0, 1.0, -10.0, -8.8)
        self.assertAlmostEqual(barrier, 1.2, places=10)

    def test_arrhenius_fit_for_tst(self):
        temps = [773.15, 823.15, 873.15, 923.15]
        row = rates_from_barrier("C_sub", 1.2, temps)
        fit = row["modified_arrhenius"]
        self.assertAlmostEqual(fit["n"], 1.0, places=6)
        self.assertAlmostEqual(fit["E_over_R_K"], ev_to_kelvin(1.2), places=4)
        self.assertAlmostEqual(fit["A"], KB_OVER_H_PER_K_S, delta=KB_OVER_H_PER_K_S * 1e-5)

    def test_branching_fraction(self):
        self.assertAlmostEqual(branching_fraction(3.0, 1.0), 0.75)

    def test_full_toy_script_outputs(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "00_run_toy_pipeline.py")],
            cwd=ROOT,
            check=True,
        )
        payload = json.loads((ROOT / "results" / "toy_arrhenius.json").read_text())
        self.assertIn("C_sub", payload)
        self.assertIn("C_int", payload)
        self.assertTrue((ROOT / "results" / "ace_surface_rates.c").exists())


if __name__ == "__main__":
    unittest.main()
