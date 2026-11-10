"""Derivative check for ComputeAlpha."""

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

from ..compute_alpha import ComputeAlpha


@pytest.mark.slow
class TestComputeAlphaDerivatives:
    """Derivative checks for the angle-of-attack component."""

    @staticmethod
    def _get_problem(low_speed_aero=False):
        ivc = om.IndepVarComp()
        polar_type = "low_speed" if low_speed_aero else "high_speed"
        ivc.add_output(f"data:aerodynamics:aircraft:{polar_type}:CL", np.linspace(0.0, 1.0, 11))
        ivc.add_output(f"data:aerodynamics:aircraft:{polar_type}:CL_alpha", 6.49, units="1/rad")
        ivc.add_output(f"data:aerodynamics:aircraft:{polar_type}:CL0", 0.1, units="unitless")
        return run_system(ComputeAlpha(low_speed_aero=low_speed_aero), ivc)

    def test_partials_high_speed(self):
        assert_component_derivatives(self._get_problem())

    def test_partials_low_speed(self):
        assert_component_derivatives(self._get_problem(low_speed_aero=True))
