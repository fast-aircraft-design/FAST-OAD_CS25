"""Derivative check for ComputeXWing."""

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

from ..geom_components.wing.components.compute_x_wing import ComputeXWing


@pytest.mark.slow
class TestComputeXWingDerivatives:
    """Derivative checks for the wing X positions component."""

    @staticmethod
    def _get_problem():
        ivc = om.IndepVarComp()
        ivc.add_output("data:geometry:wing:root:virtual_chord", 4.953, units="m")
        ivc.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
        ivc.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
        ivc.add_output("data:geometry:wing:root:y", 1.96, units="m")
        ivc.add_output("data:geometry:wing:kink:y", 6.321, units="m")
        ivc.add_output("data:geometry:wing:tip:y", 15.8015, units="m")
        ivc.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
        return run_system(ComputeXWing(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
