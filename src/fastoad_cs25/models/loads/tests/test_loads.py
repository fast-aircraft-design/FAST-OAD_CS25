"""
Test module for load functions
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2025 ONERA & ISAE-SUPAERO
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
from scipy.constants import g

from ..loads import ComputeLoads


def get_indep_var_comp(var_names):
    """Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(Path(__file__).parent / "data" / "mass_breakdown_inputs.xml")
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()
    return ivc


def test_compute_loads():
    """Tests computation of sizing loads"""
    input_list = [
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:weight:aircraft:MZFW",
        "data:weight:aircraft:MFW",
        "data:weight:aircraft:MTOW",
        "data:aerodynamics:aircraft:cruise:CL_alpha",
        "data:load_case:lc1:U_gust",
        "data:load_case:lc1:altitude",
        "data:load_case:lc1:Vc_EAS",
        "data:load_case:lc2:U_gust",
        "data:load_case:lc2:altitude",
        "data:load_case:lc2:Vc_EAS",
    ]
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:load_case:gust_intensity", val=0.5)
    problem = run_system(ComputeLoads(), ivc)

    n1m1 = problem["data:mission:sizing:cs25:sizing_load_1"]
    n2m2 = problem["data:mission:sizing:cs25:sizing_load_2"]
    n1 = problem["data:mission:sizing:cs25:load_factor_1"]
    n2 = problem["data:mission:sizing:cs25:load_factor_2"]
    assert n1m1 == pytest.approx(240968 * g, abs=10)
    assert n2m2 == pytest.approx(254130 * g, abs=10)
    assert n1 == pytest.approx(3.75, abs=0.01)
    assert n2 == pytest.approx(3.75, abs=0.01)

    # Now without fuel alleviation
    problem = run_system(ComputeLoads(fuel_load_alleviation=False), ivc)

    n1m1 = problem["data:mission:sizing:cs25:sizing_load_1"]
    n2m2 = problem["data:mission:sizing:cs25:sizing_load_2"]

    assert n1m1 == pytest.approx(240968 * g, abs=10)
    assert n2m2 == pytest.approx(284242 * g, abs=10)

    # Now with different maneuver load factor
    ivc.add_output("data:load_case:maneuver_load_factor", val=2.25)
    problem = run_system(ComputeLoads(), ivc)

    n1 = problem["data:mission:sizing:cs25:load_factor_1"]
    n2 = problem["data:mission:sizing:cs25:load_factor_2"]
    assert n1 == pytest.approx(3.375, abs=0.01)
    assert n2 == pytest.approx(3.375, abs=0.01)

    # Now without gust load alleviation
    ivc = get_indep_var_comp(input_list)
    problem = run_system(ComputeLoads(), ivc)

    n1 = problem["data:mission:sizing:cs25:load_factor_1"]
    n2 = problem["data:mission:sizing:cs25:load_factor_2"]
    n1m1 = problem["data:mission:sizing:cs25:sizing_load_1"]
    n2m2 = problem["data:mission:sizing:cs25:sizing_load_2"]
    assert n1 == pytest.approx(4.198, abs=0.01)
    assert n2 == pytest.approx(3.81, abs=0.01)
    assert n1m1 == pytest.approx(269848 * g, abs=10)
    assert n2m2 == pytest.approx(258553 * g, abs=10)
