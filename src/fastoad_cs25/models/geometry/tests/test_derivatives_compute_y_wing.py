"""Derivative check for ComputeYWing."""

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

from ..geom_components.wing.components.compute_y_wing import ComputeYWing


@pytest.mark.slow
class TestComputeYWingDerivatives:
    """Derivative checks for the wing Y positions component."""

    @staticmethod
    def _get_problem(kink_span_ratio=0.4):
        ivc = om.IndepVarComp()
        ivc.add_output("data:geometry:wing:aspect_ratio", 8.0, units="unitless")
        ivc.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
        ivc.add_output("data:geometry:wing:area", 124.843, units="m**2")
        ivc.add_output("data:geometry:wing:kink:span_ratio", kink_span_ratio, units="unitless")
        return run_system(ComputeYWing(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())

    def test_partials_without_kink(self):
        assert_component_derivatives(self._get_problem(kink_span_ratio=0.1))
