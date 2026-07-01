from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "surrogate"))

from tier0_backbone import (  # noqa: E402
    CalibrationAnchor,
    EmpiricalSiParams,
    ProcessConditions,
    ReactorGeometry,
    evaluate_profile,
    empirical_si_growth_m_s,
    torr_to_pa,
)


class Tier0SurrogateTests(unittest.TestCase):
    def test_package_import_is_dependency_light(self):
        sys.path.insert(0, str(ROOT))
        import surrogate  # noqa: import-outside-toplevel

        self.assertIn("jackel_si_space", surrogate.__all__)

    def test_empirical_si_rate_is_positive(self):
        rate = empirical_si_growth_m_s(
            923.15,
            p_dcs_Pa=1200.0,
            p_hcl_Pa=12.0,
            params=EmpiricalSiParams(),
        )
        self.assertGreater(rate, 0.0)
        self.assertTrue(math.isfinite(rate))

    def test_profile_runs_without_optional_scientific_dependencies(self):
        result = evaluate_profile(
            ProcessConditions(),
            ReactorGeometry(),
            EmpiricalSiParams(),
            CalibrationAnchor(target_si_mass_flux_kg_m2_s=7.285e-8),
            n_points=9,
        )
        self.assertEqual(len(result.profile), 9)
        self.assertGreater(result.summary["mean_growth_nm_min"], 0.0)
        self.assertIn(result.summary["dominant_regime"], {"surface-limited", "mixed", "transport-limited"})

    def test_pressure_conversion(self):
        self.assertAlmostEqual(torr_to_pa(300.0), 39996.71052631579)

    def test_run_table_manifest_uses_each_proposal_params(self):
        sys.path.insert(0, str(ROOT))
        from surrogate.runner import write_run_table  # noqa: import-outside-toplevel

        proposals = [
            {"params": {"wafer_temperature_C": 640.0, "pressure_Torr": 300.0,
                        "main_dcs_sccm": 400.0, "main_h2_sccm": 9000.0,
                        "hcl_sccm": 10.0, "rotation_rpm": 0.0},
             "fidelity": "full_chem"},
            {"params": {"wafer_temperature_C": 660.0, "pressure_Torr": 250.0,
                        "main_dcs_sccm": 450.0, "main_h2_sccm": 9500.0,
                        "hcl_sccm": 20.0, "rotation_rpm": 60.0},
             "fidelity": "flow_heat"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.csv"
            rows = write_run_table(proposals, path)
            manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
        self.assertEqual(len(rows), 2)
        self.assertEqual(manifest["run_000"]["params"]["wafer_temperature_C"], 640.0)
        self.assertEqual(manifest["run_001"]["params"]["wafer_temperature_C"], 660.0)


if __name__ == "__main__":
    unittest.main()
