"""Derivative check for ComputeHTcg."""

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

from ..cg_components.compute_ht_cg import ComputeHTcg

XML_PATH = Path(__file__).parent / "data" / "cg_inputs.xml"


@pytest.mark.slow
class TestComputeHTcgDerivatives:
    """Derivative checks for the horizontal tail CG component."""

    @staticmethod
    def _get_problem():
        reader = VariableIO(str(XML_PATH))
        reader.path_separator = ":"
        ivc = reader.read(
            only=[
                "data:geometry:horizontal_tail:center:chord",
                "data:geometry:horizontal_tail:tip:chord",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:span",
                "data:geometry:vertical_tail:span",
                "data:geometry:horizontal_tail:thickness_ratio",
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:horizontal_tail:sweep_25",
                "data:geometry:horizontal_tail:MAC:length",
                "data:geometry:fuselage:maximum_height",
                "data:geometry:has_T_tail",
            ]
        ).to_ivc()
        ivc.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", 1.656, units="m")
        return run_system(ComputeHTcg(), ivc)

    def test_partials(self):
        assert_component_derivatives(self._get_problem())

    def test_partials_with_t_tail(self):
        problem = self._get_problem()
        problem.set_val("data:geometry:has_T_tail", 1.0)
        assert_component_derivatives(problem)
