"""Derivative check for MaxCGRatiosForLoadCases."""

#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2026 ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import openmdao.api as om
import pytest
from fastoad.testing import run_system

from fastoad_cs25.models.tests.derivative_check_utils import assert_component_derivatives

from ..cg_components.load_cases.compute_cg_loadcases import MaxCGRatiosForLoadCases


@pytest.mark.slow
class TestMaxCGRatiosForLoadCasesDerivatives:
    """Derivative checks for the maximum CG ratio from load cases."""

    @staticmethod
    def _get_problem():
        ivc = om.IndepVarComp()
        ivc.add_output(
            "data:weight:aircraft:load_cases:CG:MAC_position",
            np.array([0.30, 0.40, 0.35, 0.38]),
            units="unitless",
        )
        return run_system(MaxCGRatiosForLoadCases(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
