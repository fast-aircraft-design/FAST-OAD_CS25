"""
test module for wing area computation
"""
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
from fastoad.testing import run_system
from numpy.testing import assert_allclose
from openmdao.utils.assert_utils import assert_check_partials

from ..update_wing_area_aero import UpdateWingAreaAero, WingAreaConstraintsAero


def test_update_wing_area_aero():
    ivc = om.IndepVarComp()
    ivc.add_output("data:TLAR:approach_speed", val=132, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=66300, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.8, units="unitless")
    ivc.add_output("data:geometry:wing:area", val=2.80, units="m**2")

    problem = run_system(UpdateWingAreaAero(), ivc)
    assert_allclose(problem["wing_area:aero"], 124.38, atol=1e-2)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)


def test_wing_area_constraints_aero():
    ivc = om.IndepVarComp()
    ivc.add_output("data:TLAR:approach_speed", val=132, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=66300, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.8, units="unitless")
    ivc.add_output("data:geometry:wing:area", val=116.09, units="m**2")

    problem = run_system(WingAreaConstraintsAero(), ivc)
    assert_allclose(
        problem["data:aerodynamics:aircraft:landing:additional_CL_capacity"], -0.2, atol=1e-2
    )

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)
