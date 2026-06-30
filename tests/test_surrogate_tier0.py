from __future__ import annotations

import math
import sys
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


if __name__ == "__main__":
    unittest.main()
