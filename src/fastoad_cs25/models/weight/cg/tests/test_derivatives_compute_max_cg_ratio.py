"""Derivative check for ComputeMaxCGratio."""

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

from ..cg_components.compute_max_cg_ratio import ComputeMaxCGratio

XML_PATH = Path(__file__).parent / "data" / "cg_inputs.xml"


@pytest.mark.slow
class TestComputeMaxCGratioDerivatives:
    """Derivative checks for the maximum CG ratio component."""

    @staticmethod
    def _get_problem():
        reader = VariableIO(str(XML_PATH))
        reader.path_separator = ":"
        ivc = reader.read(only=[]).to_ivc()
        ivc.add_output("data:weight:aircraft:empty:CG:MAC_position", 0.387846, units="unitless")
        ivc.add_output(
            "data:weight:aircraft:load_cases:CG:MAC_position:maximum",
            0.388971,
            units="unitless",
        )
        return run_system(ComputeMaxCGratio(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())
