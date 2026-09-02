"""
Test module for geometry functions of cg components
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

# pylint: disable=redefined-outer-name  # needed for pytest fixtures

import os.path as pth

import openmdao.api as om
import pytest
from fastoad.io import VariableIO
from fastoad.testing import run_system
from openmdao.utils.assert_utils import assert_check_partials

from ..cg import ComputeAircraftCG
from ..cg_components.compute_cg_control_surfaces import ComputeControlSurfacesCG
from ..cg_components.compute_cg_others import ComputeOthersCGX, ComputeOthersCGZ
from ..cg_components.compute_cg_ratio_aft import ComputeCGXRatioAft
from ..cg_components.compute_cg_tanks import ComputeTanksCG
from ..cg_components.compute_cg_wing import ComputeWingCG
from ..cg_components.compute_global_cg import ComputeGlobalCG
from ..cg_components.compute_ht_cg import ComputeHTcg
from ..cg_components.compute_max_cg_ratio import ComputeMaxCGratio
from ..cg_components.compute_vt_cg import ComputeVTcg
from ..cg_components.load_cases.compute_cg_loadcase1 import ComputeCGLoadCase1
from ..cg_components.load_cases.compute_cg_loadcase2 import ComputeCGLoadCase2
from ..cg_components.load_cases.compute_cg_loadcase3 import ComputeCGLoadCase3
from ..cg_components.load_cases.compute_cg_loadcase4 import ComputeCGLoadCase4
from ..cg_components.load_cases.compute_cg_loadcases import CGRatiosForLoadCases
from ..cg_components.update_mlg import UpdateMLG


@pytest.fixture(scope="module")
def input_xml() -> VariableIO:
    """
    :return: access to the sample xml data
    """
    # TODO: have more consistency in input data (no need for the whole geometry_inputs_full.xml)
    return VariableIO(pth.join(pth.dirname(__file__), "data", "cg_inputs.xml"))


def test_compute_cg_control_surfaces():
    """Tests computation of control surfaces center of gravity"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:dihedral", 6.0, units="deg")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:y", 6.293, units="m")
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.361, units="m")
    input_vars.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:kink:thickness_ratio", 0.121, units="unitless")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 2.275, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:root:thickness_ratio", 0.159, units="unitless")
    # Tip chord should not be used in this case
    input_vars.add_output("data:geometry:wing:tip:chord", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:thickness_ratio", 0.0, units="unitless")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.0, units="m")

    problem = run_system(ComputeControlSurfacesCG(), input_vars)

    x_cg_a4 = problem.get_val("data:weight:airframe:flight_controls:CG:x", units="m")
    assert x_cg_a4 == pytest.approx(19.24, abs=1e-2)

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:dihedral", 6.0, units="deg")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:y", 6.293, units="m")
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.361, units="m")
    input_vars.add_output("data:geometry:wing:kink:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:kink:thickness_ratio", 0.121, units="unitless")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:root:thickness_ratio", 0.159, units="unitless")
    # Tip chord should be used in this case
    input_vars.add_output("data:geometry:wing:tip:chord", 1.7, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 17.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:thickness_ratio", 0.11, units="unitless")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 7.8, units="m")

    problem = run_system(ComputeControlSurfacesCG(), input_vars)

    x_cg_a4 = problem.get_val("data:weight:airframe:flight_controls:CG:x", units="m")
    assert x_cg_a4 == pytest.approx(20.18, abs=1e-2)
    z_cg_a4 = problem.get_val("data:weight:airframe:flight_controls:CG:z", units="m")
    assert z_cg_a4 == pytest.approx(0.95, abs=1e-2)
    y_cg_a4 = problem.get_val("data:weight:airframe:flight_controls:CG:y", units="m")
    assert y_cg_a4 == pytest.approx(0.0, abs=1e-2)


def test_compute_cg_loadcases(input_xml):
    """Tests computation of center of gravity for load case 2"""

    input_list = [
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:MAC:at25percent:x",
        "data:weight:payload:PAX:CG:x",
        "data:weight:payload:rear_fret:CG:x",
        "data:weight:payload:front_fret:CG:x",
        "data:TLAR:NPAX",
        "data:weight:aircraft:MFW",
        "data:weight:fuel_tank:CG:x",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    input_vars.add_output("data:weight:aircraft_empty:CG:x", 699570.01 / 40979.11, units="m")
    input_vars.add_output("data:weight:aircraft_empty:mass", 40979.11, units="kg")

    # Testing each load case independently -------------
    classes = [ComputeCGLoadCase1, ComputeCGLoadCase2, ComputeCGLoadCase3, ComputeCGLoadCase4]
    cg_ratio_lc = [0.0]  # dummy first item to ensure cg_ratio_lc[i] is for load case i
    for i, cg_class in enumerate(classes):
        problem = run_system(cg_class(), input_vars)
        cg_ratio_lc.append(problem[f"data:weight:aircraft:load_case_{i + 1}:CG:MAC_position"])

    assert cg_ratio_lc[1] == pytest.approx(0.364907, abs=1e-6)
    assert cg_ratio_lc[2] == pytest.approx(0.285139, abs=1e-6)
    assert cg_ratio_lc[3] == pytest.approx(0.386260, abs=1e-6)
    assert cg_ratio_lc[4] == pytest.approx(0.388971, abs=1e-6)

    # Testing the aggregated load cases -------------
    problem = run_system(CGRatiosForLoadCases(), input_vars)
    cg_ratios = problem.get_val("data:weight:aircraft:load_cases:CG:MAC_position", units="unitless")
    max_cg_ratios = problem.get_val(
        "data:weight:aircraft:load_cases:CG:MAC_position:maximum", units="unitless"
    )

    assert cg_ratios == pytest.approx([0.364907, 0.285139, 0.386260, 0.388971], abs=1e-6)
    assert max_cg_ratios == pytest.approx(0.388971, abs=1e-6)


def test_compute_cg_x_others(input_xml):
    """Tests computation of other components center of gravity"""

    input_list = [
        "data:geometry:wing:MAC:leading_edge:x:local",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:root:chord",
        "data:geometry:fuselage:length",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:fuselage:front_length",
        "data:geometry:fuselage:rear_length",
        "data:weight:propulsion:engine:CG:x",
        "data:weight:furniture:passenger_seats:CG:x",
        "data:weight:propulsion:engine:mass",
        "data:geometry:cabin:NPAX1",
        "data:geometry:cabin:seats:economical:count_by_row",
        "data:geometry:cabin:seats:economical:length",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeOthersCGX(), input_vars)

    x_cg_a2 = problem.get_val("data:weight:airframe:fuselage:CG:x", units="m")
    assert x_cg_a2 == pytest.approx(16.88, abs=1e-2)
    x_cg_a52 = problem.get_val("data:weight:airframe:landing_gear:front:CG:x", units="m")
    assert x_cg_a52 == pytest.approx(5.18, abs=1e-2)
    x_cg_a6 = problem.get_val("data:weight:airframe:pylon:CG:x", units="m")
    assert x_cg_a6 == pytest.approx(13.5, abs=1e-1)
    x_cg_a7 = problem.get_val("data:weight:airframe:paint:CG:x", units="m")
    assert x_cg_a7 == pytest.approx(0.0, abs=1e-1)

    x_cg_b2 = problem.get_val("data:weight:propulsion:fuel_lines:CG:x", units="m")
    assert x_cg_b2 == pytest.approx(13.5, abs=1e-1)
    x_cg_b3 = problem.get_val("data:weight:propulsion:unconsumables:CG:x", units="m")
    assert x_cg_b3 == pytest.approx(13.5, abs=1e-1)

    x_cg_c11 = problem.get_val("data:weight:systems:power:auxiliary_power_unit:CG:x", units="m")
    assert x_cg_c11 == pytest.approx(35.63, abs=1e-2)
    x_cg_c12 = problem.get_val("data:weight:systems:power:electric_systems:CG:x", units="m")
    assert x_cg_c12 == pytest.approx(18.75, abs=1e-2)
    x_cg_c13 = problem.get_val("data:weight:systems:power:hydraulic_systems:CG:x", units="m")
    assert x_cg_c13 == pytest.approx(18.75, abs=1e-2)
    x_cg_c21 = problem.get_val("data:weight:systems:life_support:insulation:CG:x", units="m")
    assert x_cg_c21 == pytest.approx(16.88, abs=1e-2)
    x_cg_c22 = problem.get_val("data:weight:systems:life_support:air_conditioning:CG:x", units="m")
    assert x_cg_c22 == pytest.approx(16.62, abs=1e-2)
    x_cg_c23 = problem.get_val("data:weight:systems:life_support:de-icing:CG:x", units="m")
    assert x_cg_c23 == pytest.approx(15.79, abs=1e-2)
    x_cg_c24 = problem.get_val("data:weight:systems:life_support:cabin_lighting:CG:x", units="m")
    assert x_cg_c24 == pytest.approx(16.88, abs=1e-2)
    x_cg_c25 = problem.get_val(
        "data:weight:systems:life_support:seats_crew_accommodation:CG:x", units="m"
    )
    assert x_cg_c25 == pytest.approx(16.62, abs=1e-2)
    x_cg_c26 = problem.get_val("data:weight:systems:life_support:oxygen:CG:x", units="m")
    assert x_cg_c26 == pytest.approx(16.62, abs=1e-2)
    x_cg_c27 = problem.get_val("data:weight:systems:life_support:safety_equipment:CG:x", units="m")
    assert x_cg_c27 == pytest.approx(16.1, abs=1e-1)
    x_cg_c3 = problem.get_val("data:weight:systems:navigation:CG:x", units="m")
    assert x_cg_c3 == pytest.approx(5.52, abs=1e-2)
    x_cg_c4 = problem.get_val("data:weight:systems:transmission:CG:x", units="m")
    assert x_cg_c4 == pytest.approx(18.75, abs=1e-2)
    x_cg_c51 = problem.get_val("data:weight:systems:operational:radar:CG:x", units="m")
    assert x_cg_c51 == pytest.approx(0.75, abs=1e-2)
    x_cg_c52 = problem.get_val("data:weight:systems:operational:cargo_hold:CG:x", units="m")
    assert x_cg_c52 == pytest.approx(16.62, abs=1e-2)

    x_cg_d3 = problem.get_val("data:weight:furniture:food_water:CG:x", units="m")
    assert x_cg_d3 == pytest.approx(29.4, abs=1e-1)
    x_cg_d4 = problem.get_val("data:weight:furniture:security_kit:CG:x", units="m")
    assert x_cg_d4 == pytest.approx(16.62, abs=1e-2)
    x_cg_d5 = problem.get_val("data:weight:furniture:toilets:CG:x", units="m")
    assert x_cg_d5 == pytest.approx(16.62, abs=1e-2)
    x_cg_pl = problem.get_val("data:weight:payload:PAX:CG:x", units="m")
    assert x_cg_pl == pytest.approx(16.62, abs=1e-2)
    x_cg_rear_fret = problem.get_val("data:weight:payload:rear_fret:CG:x", units="m")
    assert x_cg_rear_fret == pytest.approx(20.87, abs=1e-2)
    x_cg_front_fret = problem.get_val("data:weight:payload:front_fret:CG:x", units="m")
    assert x_cg_front_fret == pytest.approx(9.94, abs=1e-2)


def test_compute_cg_z_others(input_xml):
    """Tests computation of other components center of gravity"""

    input_list = [
        "data:geometry:fuselage:maximum_height",
        "data:geometry:propulsion:pylon:height",
        "data:geometry:propulsion:nacelle:y",
        "data:geometry:wing:dihedral",
        "data:weight:propulsion:engine:CG:z",
        "data:weight:airframe:wing:CG:z",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeOthersCGZ(), input_vars)

    z_cg_a2 = problem.get_val("data:weight:airframe:fuselage:CG:z", units="m")
    assert z_cg_a2 == pytest.approx(2.03, abs=1e-2)
    z_cg_a52 = problem.get_val("data:weight:airframe:landing_gear:front:CG:z", units="m")
    assert z_cg_a52 == pytest.approx(0.0, abs=1e-2)
    z_cg_a6 = problem.get_val("data:weight:airframe:pylon:CG:z", units="m")
    assert z_cg_a6 == pytest.approx(0.23, abs=1e-1)
    z_cg_a7 = problem.get_val("data:weight:airframe:paint:CG:z", units="m")
    assert z_cg_a7 == pytest.approx(2.03, abs=1e-1)

    z_cg_b2 = problem.get_val("data:weight:propulsion:fuel_lines:CG:z", units="m")
    assert z_cg_b2 == pytest.approx(-0.96, abs=1e-1)
    z_cg_b3 = problem.get_val("data:weight:propulsion:unconsumables:CG:z", units="m")
    assert z_cg_b3 == pytest.approx(-0.96, abs=1e-1)

    z_cg_c11 = problem.get_val("data:weight:systems:power:auxiliary_power_unit:CG:z", units="m")
    assert z_cg_c11 == pytest.approx(2.71, abs=1e-2)
    z_cg_c12 = problem.get_val("data:weight:systems:power:electric_systems:CG:z", units="m")
    assert z_cg_c12 == pytest.approx(0.68, abs=1e-2)
    z_cg_c13 = problem.get_val("data:weight:systems:power:hydraulic_systems:CG:z", units="m")
    assert z_cg_c13 == pytest.approx(0.68, abs=1e-2)
    z_cg_c21 = problem.get_val("data:weight:systems:life_support:insulation:CG:z", units="m")
    assert z_cg_c21 == pytest.approx(2.03, abs=1e-2)
    z_cg_c22 = problem.get_val("data:weight:systems:life_support:air_conditioning:CG:z", units="m")
    assert z_cg_c22 == pytest.approx(0.68, abs=1e-2)
    z_cg_c23 = problem.get_val("data:weight:systems:life_support:de-icing:CG:z", units="m")
    assert z_cg_c23 == pytest.approx(0.86, abs=1e-2)
    z_cg_c24 = problem.get_val("data:weight:systems:life_support:cabin_lighting:CG:z", units="m")
    assert z_cg_c24 == pytest.approx(2.71, abs=1e-2)
    z_cg_c25 = problem.get_val(
        "data:weight:systems:life_support:seats_crew_accommodation:CG:z", units="m"
    )
    assert z_cg_c25 == pytest.approx(2.71, abs=1e-2)
    z_cg_c26 = problem.get_val("data:weight:systems:life_support:oxygen:CG:z", units="m")
    assert z_cg_c26 == pytest.approx(2.71, abs=1e-2)
    z_cg_c27 = problem.get_val("data:weight:systems:life_support:safety_equipment:CG:z", units="m")
    assert z_cg_c27 == pytest.approx(2.71, abs=1e-1)
    z_cg_c3 = problem.get_val("data:weight:systems:navigation:CG:z", units="m")
    assert z_cg_c3 == pytest.approx(2.71, abs=1e-2)
    z_cg_c4 = problem.get_val("data:weight:systems:transmission:CG:z", units="m")
    assert z_cg_c4 == pytest.approx(2.71, abs=1e-2)
    z_cg_c51 = problem.get_val("data:weight:systems:operational:radar:CG:z", units="m")
    assert z_cg_c51 == pytest.approx(2.03, abs=1e-2)
    z_cg_c52 = problem.get_val("data:weight:systems:operational:cargo_hold:CG:z", units="m")
    assert z_cg_c52 == pytest.approx(0.68, abs=1e-2)

    z_cg_d3 = problem.get_val("data:weight:furniture:food_water:CG:z", units="m")
    assert z_cg_d3 == pytest.approx(2.71, abs=1e-1)
    z_cg_d4 = problem.get_val("data:weight:furniture:security_kit:CG:z", units="m")
    assert z_cg_d4 == pytest.approx(2.71, abs=1e-2)
    z_cg_d5 = problem.get_val("data:weight:furniture:toilets:CG:z", units="m")
    assert z_cg_d5 == pytest.approx(2.71, abs=1e-2)


def test_compute_cg_ratio_aft(input_xml):
    """Tests computation of center of gravity with aft estimation"""

    input_list = [
        "data:weight:airframe:wing:CG:x",
        "data:weight:airframe:fuselage:CG:x",
        "data:weight:airframe:horizontal_tail:CG:x",
        "data:weight:airframe:vertical_tail:CG:x",
        "data:weight:airframe:flight_controls:CG:x",
        "data:weight:airframe:landing_gear:main:CG:x",
        "data:weight:airframe:landing_gear:front:CG:x",
        "data:weight:airframe:pylon:CG:x",
        "data:weight:airframe:paint:CG:x",
        "data:weight:airframe:wing:mass",
        "data:weight:airframe:fuselage:mass",
        "data:weight:airframe:horizontal_tail:mass",
        "data:weight:airframe:vertical_tail:mass",
        "data:weight:airframe:flight_controls:mass",
        "data:weight:airframe:landing_gear:main:mass",
        "data:weight:airframe:landing_gear:front:mass",
        "data:weight:airframe:pylon:mass",
        "data:weight:airframe:paint:mass",
        "data:weight:propulsion:engine:CG:x",
        "data:weight:propulsion:fuel_lines:CG:x",
        "data:weight:propulsion:unconsumables:CG:x",
        "data:weight:propulsion:engine:mass",
        "data:weight:propulsion:fuel_lines:mass",
        "data:weight:propulsion:unconsumables:mass",
        "data:weight:systems:power:auxiliary_power_unit:CG:x",
        "data:weight:systems:power:electric_systems:CG:x",
        "data:weight:systems:power:hydraulic_systems:CG:x",
        "data:weight:systems:life_support:insulation:CG:x",
        "data:weight:systems:life_support:air_conditioning:CG:x",
        "data:weight:systems:life_support:de-icing:CG:x",
        "data:weight:systems:life_support:cabin_lighting:CG:x",
        "data:weight:systems:life_support:seats_crew_accommodation:CG:x",
        "data:weight:systems:life_support:oxygen:CG:x",
        "data:weight:systems:life_support:safety_equipment:CG:x",
        "data:weight:systems:navigation:CG:x",
        "data:weight:systems:transmission:CG:x",
        "data:weight:systems:operational:radar:CG:x",
        "data:weight:systems:operational:cargo_hold:CG:x",
        "data:weight:systems:flight_kit:CG:x",
        "data:weight:systems:power:auxiliary_power_unit:mass",
        "data:weight:systems:power:electric_systems:mass",
        "data:weight:systems:power:hydraulic_systems:mass",
        "data:weight:systems:life_support:insulation:mass",
        "data:weight:systems:life_support:air_conditioning:mass",
        "data:weight:systems:life_support:de-icing:mass",
        "data:weight:systems:life_support:cabin_lighting:mass",
        "data:weight:systems:life_support:seats_crew_accommodation:mass",
        "data:weight:systems:life_support:oxygen:mass",
        "data:weight:systems:life_support:safety_equipment:mass",
        "data:weight:systems:navigation:mass",
        "data:weight:systems:transmission:mass",
        "data:weight:systems:operational:radar:mass",
        "data:weight:systems:operational:cargo_hold:mass",
        "data:weight:systems:flight_kit:mass",
        "data:weight:furniture:passenger_seats:CG:x",
        "data:weight:furniture:food_water:CG:x",
        "data:weight:furniture:security_kit:CG:x",
        "data:weight:furniture:toilets:CG:x",
        "data:weight:furniture:passenger_seats:mass",
        "data:weight:furniture:food_water:mass",
        "data:weight:furniture:security_kit:mass",
        "data:weight:furniture:toilets:mass",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:MAC:at25percent:x",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeCGXRatioAft(), input_vars)

    empty_mass = problem.get_val("data:weight:aircraft_empty:mass", units="kg")
    assert empty_mass == pytest.approx(41120, abs=11)
    cg_ratio_aft = problem.get_val("data:weight:aircraft:empty:CG:MAC_position", units="unitless")
    assert cg_ratio_aft == pytest.approx(0.374702, abs=1e-6)

    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=1, rtol=1e-6)


def test_compute_cg_tanks(input_xml):
    """Tests computation of tanks center of gravity"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.361, units="m")
    input_vars.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 2.275, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 7.222, units="m")
    input_vars.add_output("data:geometry:wing:spar_ratio:front:kink", 0.15, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:front:root", 0.11, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:front:tip", 0.27, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:rear:kink", 0.66, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:rear:root", 0.57, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:rear:tip", 0.56, units="unitless")

    problem = run_system(ComputeTanksCG(), input_vars)

    x_cg_tank = problem.get_val("data:weight:fuel_tank:CG:x", units="m")
    assert x_cg_tank == pytest.approx(16.12, abs=1e-2)

    # With no kink
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.361, units="m")
    input_vars.add_output("data:geometry:wing:kink:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 7.222, units="m")
    input_vars.add_output("data:geometry:wing:spar_ratio:front:kink", 0.15, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:front:root", 0.11, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:front:tip", 0.27, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:rear:kink", 0.66, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:rear:root", 0.57, units="unitless")
    input_vars.add_output("data:geometry:wing:spar_ratio:rear:tip", 0.56, units="unitless")

    problem = run_system(ComputeTanksCG(), input_vars)

    x_cg_tank = problem.get_val("data:weight:fuel_tank:CG:x", units="m")
    assert x_cg_tank == pytest.approx(16.52, abs=1e-2)


def test_compute_cg_wing(input_xml):
    """Tests computation of wing center of gravity"""

    input_list = [
        "data:geometry:wing:dihedral",
        "data:geometry:wing:kink:span_ratio",
        "data:geometry:wing:spar_ratio:front:root",
        "data:geometry:wing:spar_ratio:front:kink",
        "data:geometry:wing:spar_ratio:front:tip",
        "data:geometry:wing:spar_ratio:rear:root",
        "data:geometry:wing:spar_ratio:rear:kink",
        "data:geometry:wing:spar_ratio:rear:tip",
        "data:geometry:wing:span",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:MAC:leading_edge:x:local",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:kink:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:root:y",
        "data:geometry:wing:root:thickness_ratio",
        "data:geometry:wing:kink:leading_edge:x:local",
        "data:geometry:wing:kink:y",
        "data:geometry:wing:kink:thickness_ratio",
        "data:geometry:wing:tip:y",
        "data:geometry:wing:tip:leading_edge:x:local",
        "data:geometry:wing:tip:thickness_ratio",
        "data:geometry:wing:MAC:at25percent:x",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeWingCG(), input_vars)

    x_cg_wing = problem.get_val("data:weight:airframe:wing:CG:x", units="m")
    assert x_cg_wing == pytest.approx(16.67, abs=1e-2)
    z_cg_wing = problem.get_val("data:weight:airframe:wing:CG:z", units="m")
    assert z_cg_wing == pytest.approx(0.86, abs=1e-2)
    y_cg_wing = problem.get_val("data:weight:airframe:wing:CG:y", units="m")
    assert y_cg_wing == pytest.approx(0.0, abs=1e-2)


def test_compute_global_cg(input_xml):
    """Tests computation of global center of gravity"""

    input_list = [
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:MAC:at25percent:x",
        "data:weight:payload:PAX:CG:x",
        "data:weight:payload:rear_fret:CG:x",
        "data:weight:payload:front_fret:CG:x",
        "data:TLAR:NPAX",
        "data:weight:aircraft:MFW",
        "data:weight:fuel_tank:CG:x",
        "data:weight:airframe:wing:CG:x",
        "data:weight:airframe:fuselage:CG:x",
        "data:weight:airframe:horizontal_tail:CG:x",
        "data:weight:airframe:vertical_tail:CG:x",
        "data:weight:airframe:flight_controls:CG:x",
        "data:weight:airframe:landing_gear:main:CG:x",
        "data:weight:airframe:landing_gear:front:CG:x",
        "data:weight:airframe:pylon:CG:x",
        "data:weight:airframe:paint:CG:x",
        "data:weight:airframe:wing:mass",
        "data:weight:airframe:fuselage:mass",
        "data:weight:airframe:horizontal_tail:mass",
        "data:weight:airframe:vertical_tail:mass",
        "data:weight:airframe:flight_controls:mass",
        "data:weight:airframe:landing_gear:main:mass",
        "data:weight:airframe:landing_gear:front:mass",
        "data:weight:airframe:pylon:mass",
        "data:weight:airframe:paint:mass",
        "data:weight:propulsion:engine:CG:x",
        "data:weight:propulsion:fuel_lines:CG:x",
        "data:weight:propulsion:unconsumables:CG:x",
        "data:weight:propulsion:engine:mass",
        "data:weight:propulsion:fuel_lines:mass",
        "data:weight:propulsion:unconsumables:mass",
        "data:weight:systems:power:auxiliary_power_unit:CG:x",
        "data:weight:systems:power:electric_systems:CG:x",
        "data:weight:systems:power:hydraulic_systems:CG:x",
        "data:weight:systems:life_support:insulation:CG:x",
        "data:weight:systems:life_support:air_conditioning:CG:x",
        "data:weight:systems:life_support:de-icing:CG:x",
        "data:weight:systems:life_support:cabin_lighting:CG:x",
        "data:weight:systems:life_support:seats_crew_accommodation:CG:x",
        "data:weight:systems:life_support:oxygen:CG:x",
        "data:weight:systems:life_support:safety_equipment:CG:x",
        "data:weight:systems:navigation:CG:x",
        "data:weight:systems:transmission:CG:x",
        "data:weight:systems:operational:radar:CG:x",
        "data:weight:systems:operational:cargo_hold:CG:x",
        "data:weight:systems:flight_kit:CG:x",
        "data:weight:systems:power:auxiliary_power_unit:mass",
        "data:weight:systems:power:electric_systems:mass",
        "data:weight:systems:power:hydraulic_systems:mass",
        "data:weight:systems:life_support:insulation:mass",
        "data:weight:systems:life_support:air_conditioning:mass",
        "data:weight:systems:life_support:de-icing:mass",
        "data:weight:systems:life_support:cabin_lighting:mass",
        "data:weight:systems:life_support:seats_crew_accommodation:mass",
        "data:weight:systems:life_support:oxygen:mass",
        "data:weight:systems:life_support:safety_equipment:mass",
        "data:weight:systems:navigation:mass",
        "data:weight:systems:transmission:mass",
        "data:weight:systems:operational:radar:mass",
        "data:weight:systems:operational:cargo_hold:mass",
        "data:weight:systems:flight_kit:mass",
        "data:weight:furniture:passenger_seats:CG:x",
        "data:weight:furniture:food_water:CG:x",
        "data:weight:furniture:security_kit:CG:x",
        "data:weight:furniture:toilets:CG:x",
        "data:weight:furniture:passenger_seats:mass",
        "data:weight:furniture:food_water:mass",
        "data:weight:furniture:security_kit:mass",
        "data:weight:furniture:toilets:mass",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeGlobalCG(), input_vars)

    cg_ratio = problem.get_val("data:weight:aircraft:CG:aft:MAC_position", units="unitless")
    assert cg_ratio == pytest.approx(0.430052, abs=1e-6)


def test_compute_max_cg_ratio(input_xml):
    """Tests computation of maximum center of gravity ratio"""

    input_list = []

    input_vars = input_xml.read(only=input_list).to_ivc()

    input_vars.add_output("data:weight:aircraft:empty:CG:MAC_position", 0.387846, units="unitless")
    input_vars.add_output(
        "data:weight:aircraft:load_cases:CG:MAC_position:maximum", 0.388971, units="unitless"
    )

    problem = run_system(ComputeMaxCGratio(), input_vars)

    cg_ratio = problem.get_val("data:weight:aircraft:CG:aft:MAC_position", units="unitless")
    assert cg_ratio == pytest.approx(0.438971, abs=1e-6)


def test_compute_aircraft_cg(input_xml):
    """Tests computation of static margin"""

    input_list = ["data:geometry:wing:MAC:length", "data:geometry:wing:MAC:at25percent:x"]

    input_vars = input_xml.read(only=input_list).to_ivc()
    input_vars.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.388971, units="unitless")

    problem = run_system(ComputeAircraftCG(), input_vars)

    cg_global = problem.get_val("data:weight:aircraft:CG:aft:x", units="m")
    assert cg_global == pytest.approx(17.1, abs=1e-1)


def test_compute_ht_cg(input_xml):
    """Tests computation of the horizontal tail center of gravity"""

    input_list = [
        "data:geometry:horizontal_tail:center:chord",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:horizontal_tail:span",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:MAC:length",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()
    input_vars.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", 1.656, units="m")

    problem = run_system(ComputeHTcg(), input_vars)

    cg_a31 = problem.get_val("data:weight:airframe:horizontal_tail:CG:x", units="m")
    assert cg_a31 == pytest.approx(34.58, abs=1e-2)


def test_compute_vt_cg(input_xml):
    """Tests computation of the vertical tail center of gravity"""

    input_list = [
        "data:geometry:vertical_tail:root:chord",
        "data:geometry:vertical_tail:tip:chord",
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:vertical_tail:span",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:vertical_tail:MAC:length",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    input_vars.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", 2.321, units="m")

    component = ComputeVTcg()

    problem = run_system(component, input_vars)

    cg_a32 = problem.get_val("data:weight:airframe:vertical_tail:CG:x", units="m")
    assert cg_a32 == pytest.approx(34.265, abs=1e-3)


def test_geometry_update_mlg(input_xml):
    """Tests computation of the main landing gear"""

    input_list = ["data:geometry:wing:MAC:length", "data:geometry:wing:MAC:at25percent:x"]

    input_vars = input_xml.read(only=input_list).to_ivc()

    input_vars.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.364924, units="unitless")
    input_vars.add_output("data:weight:airframe:landing_gear:front:CG:x", 5.07, units="m")

    problem = run_system(UpdateMLG(), input_vars)

    cg_a51 = problem.get_val("data:weight:airframe:landing_gear:main:CG:x", units="m")
    assert cg_a51 == pytest.approx(18.00, abs=1e-2)
