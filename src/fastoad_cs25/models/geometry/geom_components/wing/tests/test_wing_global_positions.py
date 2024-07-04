#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2024 ONERA & ISAE-SUPAERO
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

from ..wing_global_positions import ComputeChordGlobalPositions


def test_chord_global_positions():
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 10.0, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.0, units="m")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.0, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 2.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 5.5, units="m")

    problem = run_system(ComputeChordGlobalPositions(), input_vars)
    assert problem["data:geometry:wing:root:leading_edge:x"] == pytest.approx(7.0, abs=1e-2)
    assert problem["data:geometry:wing:kink:leading_edge:x"] == pytest.approx(9.0, abs=1e-2)
    assert problem["data:geometry:wing:tip:leading_edge:x"] == pytest.approx(12.5, abs=1e-2)
