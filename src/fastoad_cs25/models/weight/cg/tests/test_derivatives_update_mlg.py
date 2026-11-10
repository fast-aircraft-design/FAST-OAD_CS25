"""Derivative check for UpdateMLG."""

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
from fastoad.io import VariableIO
from fastoad.testing import run_system

from fastoad_cs25.models.tests.derivative_check_utils import assert_component_derivatives

from ..cg_components.update_mlg import UpdateMLG

XML_PATH = Path(__file__).parent / "data" / "cg_inputs.xml"


@pytest.mark.slow
class TestUpdateMLGDerivatives:
    """Derivative checks for the main landing gear CG component."""

    @staticmethod
    def _get_problem():
        reader = VariableIO(str(XML_PATH))
        reader.path_separator = ":"
        ivc = reader.read(
            only=["data:geometry:wing:MAC:length", "data:geometry:wing:MAC:at25percent:x"]
        ).to_ivc()
        ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.364924, units="unitless")
        ivc.add_output("data:weight:airframe:landing_gear:front:CG:x", 5.07, units="m")
        return run_system(UpdateMLG(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
