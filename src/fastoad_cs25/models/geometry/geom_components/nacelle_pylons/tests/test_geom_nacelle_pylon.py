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

import openmdao.api as om
import pytest
from fastoad.testing import run_system

from ..compute_nacelle_pylons import ComputeNacelleAndPylonsGeometry


def test_geometry_nacelle_pylons():
    """Tests computation of the nacelle and pylons component"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:length", 37.507, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:propulsion:layout", 1.0, units="unitless")
    input_vars.add_output("data:geometry:propulsion:engine:y_ratio", 0.34, units="unitless")
    input_vars.add_output("data:geometry:wing:span", 31.603, units="m")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.361, units="m")
    input_vars.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 2.275, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:propulsion:MTO_thrust", 117880.0, units="N")
    # Tip chord should not be used in this case
    input_vars.add_output("data:geometry:wing:tip:chord", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.0, units="m")

    component = ComputeNacelleAndPylonsGeometry()

    problem = run_system(component, input_vars)

    pylon_length = problem["data:geometry:propulsion:pylon:length"]
    assert pylon_length == pytest.approx(5.733, abs=1e-3)
    fan_length = problem["data:geometry:propulsion:fan:length"]
    assert fan_length == pytest.approx(3.127, abs=1e-3)
    nacelle_length = problem["data:geometry:propulsion:nacelle:length"]
    assert nacelle_length == pytest.approx(5.211, abs=1e-3)
    nacelle_dia = problem["data:geometry:propulsion:nacelle:diameter"]
    assert nacelle_dia == pytest.approx(2.172, abs=1e-3)
    lg_height = problem["data:geometry:landing_gear:height"]
    assert lg_height == pytest.approx(3.041, abs=1e-3)
    y_nacell = problem["data:geometry:propulsion:nacelle:y"]
    assert y_nacell == pytest.approx(5.373, abs=1e-3)
    pylon_wet_area = problem["data:geometry:propulsion:pylon:wetted_area"]
    assert pylon_wet_area == pytest.approx(7.563, abs=1e-3)
    nacelle_wet_area = problem["data:geometry:propulsion:nacelle:wetted_area"]
    assert nacelle_wet_area == pytest.approx(21.609, abs=1e-3)
    cg_b1 = problem["data:weight:propulsion:engine:CG:x"]
    assert cg_b1 == pytest.approx(13.5, abs=1e-1)


def test_geometry_nacelle_pylons_absolute_engine_y():
    """Tests computation of the nacelle and pylons component"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:length", 37.507, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:propulsion:layout", 1.0, units="unitless")
    input_vars.add_output("data:geometry:propulsion:nacelle:y", 5.373, units="m")
    input_vars.add_output("data:geometry:wing:span", 31.603, units="m")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.361, units="m")
    input_vars.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 2.275, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:propulsion:MTO_thrust", 117880.0, units="N")
    # Tip chord should not be used in this case
    input_vars.add_output("data:geometry:wing:tip:chord", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.0, units="m")

    component = ComputeNacelleAndPylonsGeometry(impose_absolute_engine=True)

    problem = run_system(component, input_vars)

    pylon_length = problem["data:geometry:propulsion:pylon:length"]
    assert pylon_length == pytest.approx(5.733, abs=1e-3)
    fan_length = problem["data:geometry:propulsion:fan:length"]
    assert fan_length == pytest.approx(3.127, abs=1e-3)
    nacelle_length = problem["data:geometry:propulsion:nacelle:length"]
    assert nacelle_length == pytest.approx(5.211, abs=1e-3)
    nacelle_dia = problem["data:geometry:propulsion:nacelle:diameter"]
    assert nacelle_dia == pytest.approx(2.172, abs=1e-3)
    lg_height = problem["data:geometry:landing_gear:height"]
    assert lg_height == pytest.approx(3.041, abs=1e-3)
    y_nacelle_ratio = problem["data:geometry:propulsion:engine:y_ratio"]
    assert y_nacelle_ratio == pytest.approx(0.34, abs=1e-3)
    pylon_wet_area = problem["data:geometry:propulsion:pylon:wetted_area"]
    assert pylon_wet_area == pytest.approx(7.563, abs=1e-3)
    nacelle_wet_area = problem["data:geometry:propulsion:nacelle:wetted_area"]
    assert nacelle_wet_area == pytest.approx(21.609, abs=1e-3)
    cg_b1 = problem["data:weight:propulsion:engine:CG:x"]
    assert cg_b1 == pytest.approx(13.5, abs=1e-1)


def test_geometry_nacelle_pylons_no_kink():
    """Tests computation of the nacelle and pylons component without kink"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:length", 37.507, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:propulsion:layout", 1.0, units="unitless")
    input_vars.add_output("data:geometry:propulsion:engine:y_ratio", 0.34, units="unitless")
    input_vars.add_output("data:geometry:wing:span", 31.603, units="m")
    input_vars.add_output("data:geometry:wing:MAC:length", 4.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:at25percent:x", 16.457, units="m")
    input_vars.add_output("data:geometry:wing:MAC:leading_edge:x:local", 2.361, units="m")
    input_vars.add_output("data:geometry:wing:kink:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 0.0, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:propulsion:MTO_thrust", 117880.0, units="N")
    # Tip chord should be used in this case
    input_vars.add_output("data:geometry:wing:tip:chord", 1.7, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 17.0, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 7.8, units="m")

    component = ComputeNacelleAndPylonsGeometry()

    problem = run_system(component, input_vars)

    pylon_length = problem["data:geometry:propulsion:pylon:length"]
    assert pylon_length == pytest.approx(5.733, abs=1e-3)
    fan_length = problem["data:geometry:propulsion:fan:length"]
    assert fan_length == pytest.approx(3.127, abs=1e-3)
    nacelle_length = problem["data:geometry:propulsion:nacelle:length"]
    assert nacelle_length == pytest.approx(5.211, abs=1e-3)
    nacelle_dia = problem["data:geometry:propulsion:nacelle:diameter"]
    assert nacelle_dia == pytest.approx(2.172, abs=1e-3)
    lg_height = problem["data:geometry:landing_gear:height"]
    assert lg_height == pytest.approx(3.041, abs=1e-3)
    y_nacell = problem["data:geometry:propulsion:nacelle:y"]
    assert y_nacell == pytest.approx(5.373, abs=1e-3)
    pylon_wet_area = problem["data:geometry:propulsion:pylon:wetted_area"]
    assert pylon_wet_area == pytest.approx(7.563, abs=1e-3)
    nacelle_wet_area = problem["data:geometry:propulsion:nacelle:wetted_area"]
    assert nacelle_wet_area == pytest.approx(21.609, abs=1e-3)
    cg_b1 = problem["data:weight:propulsion:engine:CG:x"]
    assert cg_b1 == pytest.approx(13.5, abs=1e-1)
