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

from ..update_wing_area_geom import UpdateWingAreaGeom, WingAreaConstraintsGeom


def test_update_wing_area_geom():
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:aspect_ratio", 9.48, units="unitless")
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.15, units="unitless")
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.11, units="unitless")
    ivc.add_output("data:weight:aircraft:sizing_block_fuel", val=20500, units="kg")

    problem = run_system(UpdateWingAreaGeom(), ivc)
    assert_allclose(problem["wing_area:geom"], 133.97, atol=1e-2)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)


def test_wing_area_constraints_geom():
    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:aircraft:sizing_block_fuel", val=15000.0, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=12000.0, units="kg")

    problem = run_system(WingAreaConstraintsGeom(), ivc)
    assert_allclose(problem["data:weight:aircraft:additional_fuel_capacity"], -3000.0, atol=1e-2)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)
