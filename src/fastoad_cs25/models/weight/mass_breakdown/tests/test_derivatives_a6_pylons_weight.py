"""Derivative check for PylonsWeight."""

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

from pathlib import Path

import pytest
from fastoad.testing import run_system

from fastoad_cs25.models.tests.derivative_check_utils import (
    assert_component_derivatives,
    get_ivc_from_xml,
)

from ..a_airframe.a6_pylons_weight import PylonsWeight

XML_PATH = Path(__file__).parent / "data" / "mass_breakdown_inputs.xml"


@pytest.mark.slow
class TestPylonsWeightDerivatives:
    """Derivative checks for the pylons weight component."""

    @staticmethod
    def _get_problem():
        ivc = get_ivc_from_xml(XML_PATH)
        ivc.add_output("data:weight:propulsion:engine:mass", 7161.33, units="kg")
        return run_system(PylonsWeight(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())

    def test_partials_with_engines_on_fuselage(self):
        problem = self._get_problem()
        problem.set_val("data:geometry:propulsion:layout", 2.0)
        assert_component_derivatives(problem)
