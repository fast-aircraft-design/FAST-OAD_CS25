"""Derivative check for PassengerSeatsWeight."""

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

from ..d_furniture.d2_passenger_seats_weight import PassengerSeatsWeight

XML_PATH = Path(__file__).parent / "data" / "mass_breakdown_inputs.xml"


@pytest.mark.slow
class TestPassengerSeatsWeightDerivatives:
    """Derivative checks for the passenger seats weight component."""

    @staticmethod
    def _get_problem():
        ivc = get_ivc_from_xml(XML_PATH)
        return run_system(PassengerSeatsWeight(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())

    def test_partials_for_other_range_categories(self):
        problem = self._get_problem()
        for tlar_range in [500.0, 2000.0, 3500.0, 5000.0, 7000.0]:
            problem.set_val("data:TLAR:range", tlar_range, units="NM")
            assert_component_derivatives(problem)
