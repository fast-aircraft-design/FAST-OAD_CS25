"""Derivative check for ComputeWingPosition."""

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

import openmdao.api as om
import pytest
from fastoad.testing import run_system

from fastoad_cs25.models.tests.derivative_check_utils import assert_component_derivatives

from ..compute_wing_position import ComputeWingPosition


@pytest.mark.slow
class TestComputeWingPositionDerivatives:
    """Derivative checks for the wing position component."""

    @staticmethod
    def _get_problem():
        ivc = om.IndepVarComp()
        ivc.add_output("data:handling_qualities:static_margin", 0.1, units="unitless")
        ivc.add_output("data:handling_qualities:static_margin:target", 0.05, units="unitless")
        ivc.add_output("data:geometry:wing:MAC:length", 4.262, units="m")
        ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.361, units="unitless")
        ivc.add_output("data:weight:aircraft:CG:aft:x", 17.545, units="m")
        return run_system(ComputeWingPosition(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
