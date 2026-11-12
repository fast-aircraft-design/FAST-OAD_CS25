"""Derivative check for ComputeCGLoadCase."""

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

from ..cg_components.load_cases.compute_cg_loadcase_base import ComputeCGLoadCase


@pytest.mark.slow
class TestComputeCGLoadCaseDerivatives:
    """Derivative checks for the generic CG load case component."""

    @staticmethod
    def _get_problem():
        ivc = om.IndepVarComp()
        ivc.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
        ivc.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
        ivc.add_output("data:weight:payload:PAX:CG:x", 15.0, units="m")
        ivc.add_output("data:weight:payload:rear_fret:CG:x", 20.0, units="m")
        ivc.add_output("data:weight:payload:front_fret:CG:x", 10.0, units="m")
        ivc.add_output("data:TLAR:NPAX", 150.0, units="unitless")
        ivc.add_output("data:weight:aircraft_empty:CG:x", 16.0, units="m")
        ivc.add_output("data:weight:aircraft_empty:mass", 40000.0, units="kg")
        return run_system(ComputeCGLoadCase(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
