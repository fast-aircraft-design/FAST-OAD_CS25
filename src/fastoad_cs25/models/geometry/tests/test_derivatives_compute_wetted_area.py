"""Derivative check for ComputeWettedArea."""

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

from ..geom_components.compute_wetted_area import ComputeWettedArea


@pytest.mark.slow
class TestComputeWettedAreaDerivatives:
    """Derivative checks for the aircraft wetted area component."""

    @staticmethod
    def _get_problem():
        ivc = om.IndepVarComp()
        ivc.add_output("data:geometry:wing:wetted_area", 200.0, units="m**2")
        ivc.add_output("data:geometry:fuselage:wetted_area", 401.956, units="m**2")
        ivc.add_output("data:geometry:horizontal_tail:wetted_area", 35.0, units="m**2")
        ivc.add_output("data:geometry:vertical_tail:wetted_area", 25.0, units="m**2")
        ivc.add_output("data:geometry:propulsion:nacelle:wetted_area", 20.0, units="m**2")
        ivc.add_output("data:geometry:propulsion:pylon:wetted_area", 7.563, units="m**2")
        ivc.add_output("data:geometry:propulsion:engine:count", 2.0, units="unitless")
        return run_system(ComputeWettedArea(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
