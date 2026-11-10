"""Derivative check for FuselageWeight."""

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
from scipy.constants import g

from fastoad_cs25.models.tests.derivative_check_utils import (
    assert_component_derivatives,
    get_ivc_from_xml,
)

from ..a_airframe.a2_fuselage_weight import FuselageWeight

XML_PATH = Path(__file__).parent / "data" / "mass_breakdown_inputs.xml"


@pytest.mark.slow
class TestFuselageWeightDerivatives:
    """Derivative checks for the fuselage weight component."""

    @staticmethod
    def _get_problem():
        ivc = get_ivc_from_xml(XML_PATH)
        ivc.add_output("data:mission:sizing:cs25:envelope:max_sizing_load_1", 241000 * g, units="N")
        return run_system(FuselageWeight(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
