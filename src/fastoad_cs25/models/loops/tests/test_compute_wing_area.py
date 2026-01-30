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
import pytest
from fastoad.testing import run_system
from numpy.testing import assert_allclose
from openmdao.utils.assert_utils import assert_check_partials

from ..compute_wing_area import ComputeWingArea


@pytest.fixture(scope="session")
def base_ivc():
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:aspect_ratio", 9.48, units="unitless")
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.15, units="unitless")
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.11, units="unitless")
    ivc.add_output("data:weight:aircraft:sizing_block_fuel", val=20500, units="kg")
    ivc.add_output("data:TLAR:approach_speed", val=132, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=66300, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=21000, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.80, units="unitless")

    return ivc


def test_compute_wing_area(base_ivc):
    # Driven by fuel ===========================================================
    problem = run_system(ComputeWingArea(), base_ivc)

    assert_allclose(problem["data:geometry:wing:area"], 133.97, atol=1e-2)
    assert_allclose(
        problem["data:aerodynamics:aircraft:landing:additional_CL_capacity"], 0.199, atol=1e-2
    )
    assert_allclose(problem["data:weight:aircraft:additional_fuel_capacity"], 500.0, atol=1.0)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)

    # Driven by CL max =========================================================
    problem.set_val("data:weight:aircraft:sizing_block_fuel", val=15000, units="kg")
    problem.run_model()

    assert_allclose(problem["data:geometry:wing:area"], 124.38, atol=1e-2)
    assert_allclose(
        problem["data:aerodynamics:aircraft:landing:additional_CL_capacity"], 0.0, atol=1e-2
    )
    assert_allclose(problem["data:weight:aircraft:additional_fuel_capacity"], 6000.0, atol=1.0)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)


def test_compute_wing_area_using_only_approach_speed(base_ivc):
    # Driven by fuel ===========================================================
    problem = run_system(ComputeWingArea(use_fuel=False), base_ivc)

    assert_allclose(problem["data:geometry:wing:area"], 124.38, atol=1e-2)
    assert_allclose(
        problem["data:aerodynamics:aircraft:landing:additional_CL_capacity"], 0.0, atol=1e-2
    )
    assert_allclose(problem["data:weight:aircraft:additional_fuel_capacity"], 500.0, atol=1.0)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)

    # Driven by CL max =========================================================
    problem.set_val("data:weight:aircraft:sizing_block_fuel", val=30000, units="kg")
    problem.run_model()

    assert_allclose(problem["data:geometry:wing:area"], 124.38, atol=1e-2)
    assert_allclose(
        problem["data:aerodynamics:aircraft:landing:additional_CL_capacity"], 0.0, atol=1e-2
    )
    assert_allclose(problem["data:weight:aircraft:additional_fuel_capacity"], -9000.0, atol=1.0)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)


def test_compute_wing_area_using_only_fuel(base_ivc):
    # Driven by fuel ===========================================================
    problem = run_system(ComputeWingArea(use_approach_speed=False), base_ivc)

    assert_allclose(problem["data:geometry:wing:area"], 133.97, atol=1e-2)
    assert_allclose(
        problem["data:aerodynamics:aircraft:landing:additional_CL_capacity"], 0.199, atol=1e-2
    )
    assert_allclose(problem["data:weight:aircraft:additional_fuel_capacity"], 500.0, atol=1.0)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)

    # Driven by CL max =========================================================
    problem.set_val("data:weight:aircraft:sizing_block_fuel", val=15000, units="kg")
    problem.run_model()

    assert_allclose(problem["data:geometry:wing:area"], 106.57, atol=1e-2)
    assert_allclose(
        problem["data:aerodynamics:aircraft:landing:additional_CL_capacity"], -0.47, atol=1e-2
    )
    assert_allclose(problem["data:weight:aircraft:additional_fuel_capacity"], 6000.0, atol=1.0)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-4)
