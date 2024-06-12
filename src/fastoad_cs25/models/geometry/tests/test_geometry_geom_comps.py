"""
Test module for geometry functions of cg components
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2022 ONERA & ISAE-SUPAERO
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

import pytest
from fastoad._utils.testing import run_system
from fastoad.io import VariableIO

from ..geom_components.compute_wetted_area import ComputeWettedArea
from ..geom_components.fuselage.compute_cnbeta_fuselage import ComputeCnBetaFuselage
from ..geom_components.fuselage.compute_fuselage import (
    ComputeFuselageGeometryBasic,
    ComputeFuselageGeometryCabinSizing,
)
from ..geom_components.ht.components import (
    ComputeHTChord,
    ComputeHTClalpha,
    ComputeHTMAC,
    ComputeHTSweep,
)
from ..geom_components.vt.components import (
    ComputeVTChords,
    ComputeVTClalpha,
    ComputeVTDistance,
    ComputeVTMAC,
    ComputeVTSweep,
    ComputeVTLocalPositions,
)

from ..geom_components.vt.vt_global_positions import ComputeChordGlobalPositions


@pytest.fixture(scope="module")
def input_xml() -> VariableIO:
    """
    :return: access to the sample xml data
    """
    # TODO: have more consistency in input data (no need for the whole geometry_inputs_full.xml)
    return VariableIO(pth.join(pth.dirname(__file__), "data", "geometry_inputs_full.xml"))


def test_compute_fuselage_cabin_sizing(input_xml):
    """Tests computation of the fuselage with cabin sizing"""

    input_list = [
        "data:geometry:cabin:seats:economical:width",
        "data:geometry:cabin:seats:economical:length",
        "data:geometry:cabin:seats:economical:count_by_row",
        "data:geometry:cabin:aisle_width",
        "data:geometry:cabin:exit_width",
        "data:TLAR:NPAX",
        "data:geometry:propulsion:engine:count",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeFuselageGeometryCabinSizing(), input_vars)

    npax1 = problem["data:geometry:cabin:NPAX1"]
    assert npax1 == pytest.approx(157, abs=1)
    cg_systems_c6 = problem["data:weight:systems:flight_kit:CG:x"]
    assert cg_systems_c6 == pytest.approx(7.47, abs=1e-2)
    cg_furniture_d2 = problem["data:weight:furniture:passenger_seats:CG:x"]
    assert cg_furniture_d2 == pytest.approx(16.62, abs=1e-2)
    fuselage_length = problem["data:geometry:fuselage:length"]
    assert fuselage_length == pytest.approx(37.507, abs=1e-3)
    fuselage_width_max = problem["data:geometry:fuselage:maximum_width"]
    assert fuselage_width_max == pytest.approx(3.92, abs=1e-2)
    fuselage_height_max = problem["data:geometry:fuselage:maximum_height"]
    assert fuselage_height_max == pytest.approx(4.06, abs=1e-2)
    fuselage_lav = problem["data:geometry:fuselage:front_length"]
    assert fuselage_lav == pytest.approx(6.902, abs=1e-3)
    fuselage_lar = problem["data:geometry:fuselage:rear_length"]
    assert fuselage_lar == pytest.approx(14.616, abs=1e-3)
    fuselage_lpax = problem["data:geometry:fuselage:PAX_length"]
    assert fuselage_lpax == pytest.approx(22.87, abs=1e-2)
    fuselage_lcabin = problem["data:geometry:cabin:length"]
    assert fuselage_lcabin == pytest.approx(30.38, abs=1e-2)
    fuselage_wet_area = problem["data:geometry:fuselage:wetted_area"]
    assert fuselage_wet_area == pytest.approx(401.956, abs=1e-3)
    pnc = problem["data:geometry:cabin:crew_count:commercial"]
    assert pnc == pytest.approx(4, abs=1)


def test_compute_fuselage_basic(input_xml):
    """Tests computation of the fuselage with no cabin sizing"""

    input_list = [
        "data:geometry:cabin:NPAX1",
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:front_length",
        "data:geometry:fuselage:rear_length",
        "data:geometry:fuselage:PAX_length",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeFuselageGeometryBasic(), input_vars)

    cg_systems_c6 = problem["data:weight:systems:flight_kit:CG:x"]
    assert cg_systems_c6 == pytest.approx(9.19, abs=1e-2)
    cg_furniture_d2 = problem["data:weight:furniture:passenger_seats:CG:x"]
    assert cg_furniture_d2 == pytest.approx(14.91, abs=1e-2)
    fuselage_lcabin = problem["data:geometry:cabin:length"]
    assert fuselage_lcabin == pytest.approx(30.38, abs=1e-2)
    fuselage_wet_area = problem["data:geometry:fuselage:wetted_area"]
    assert fuselage_wet_area == pytest.approx(401.962, abs=1e-3)
    pnc = problem["data:geometry:cabin:crew_count:commercial"]
    assert pnc == pytest.approx(4, abs=1)


def test_compute_ht_mac(input_xml):
    """Tests computation of the horizontal tail mac"""

    input_list = [
        "data:geometry:horizontal_tail:root:chord",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:horizontal_tail:span",
        "data:geometry:horizontal_tail:sweep_25",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeHTMAC(), input_vars)

    length = problem["data:geometry:horizontal_tail:MAC:length"]
    assert length == pytest.approx(3.141, abs=1e-3)
    ht_x0 = problem["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
    assert ht_x0 == pytest.approx(1.656, abs=1e-3)
    ht_y0 = problem["data:geometry:horizontal_tail:MAC:y"]
    assert ht_y0 == pytest.approx(2.519, abs=1e-3)


def test_compute_ht_chord(input_xml):
    """Tests computation of the horizontal tail chords"""

    input_list = [
        "data:geometry:horizontal_tail:aspect_ratio",
        "data:geometry:horizontal_tail:area",
        "data:geometry:horizontal_tail:taper_ratio",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeHTChord(), input_vars)

    span = problem["data:geometry:horizontal_tail:span"]
    assert span == pytest.approx(12.28, abs=1e-2)
    root_chord = problem["data:geometry:horizontal_tail:root:chord"]
    assert root_chord == pytest.approx(4.406, abs=1e-3)
    tip_chord = problem["data:geometry:horizontal_tail:tip:chord"]
    assert tip_chord == pytest.approx(1.322, abs=1e-3)


def test_compute_ht_cl(input_xml):
    """Tests computation of the horizontal tail lift coefficient"""

    input_list = [
        "data:geometry:horizontal_tail:aspect_ratio",
        "data:TLAR:cruise_mach",
        "data:geometry:horizontal_tail:sweep_25",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeHTClalpha(), input_vars)

    cl_alpha = problem["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
    assert cl_alpha == pytest.approx(3.47, abs=1e-2)


def test_compute_ht_sweep(input_xml):
    """Tests computation of the horizontal tail sweep"""

    input_list = [
        "data:geometry:horizontal_tail:root:chord",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:horizontal_tail:span",
        "data:geometry:horizontal_tail:sweep_25",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    problem = run_system(ComputeHTSweep(), input_vars)

    sweep_0 = problem["data:geometry:horizontal_tail:sweep_0"]
    assert sweep_0 == pytest.approx(33.317, abs=1e-3)
    sweep_100 = problem["data:geometry:horizontal_tail:sweep_100"]
    assert sweep_100 == pytest.approx(8.81, abs=1e-2)


def test_compute_fuselage_cnbeta(input_xml):
    """Tests computation of the yawing moment due to sideslip"""

    input_list = [
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:front_length",
        "data:geometry:fuselage:rear_length",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    component = ComputeCnBetaFuselage()

    problem = run_system(component, input_vars)

    cn_beta = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta == pytest.approx(-0.117901, abs=1e-6)


def test_compute_vt_mac(input_xml):
    """Tests computation of the vertical tail mac"""

    input_list = [
        "data:geometry:vertical_tail:root:chord",
        "data:geometry:vertical_tail:tip:chord",
        "data:geometry:vertical_tail:span",
        "data:geometry:vertical_tail:sweep_25",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    component = ComputeVTMAC()

    problem = run_system(component, input_vars)

    length = problem["data:geometry:vertical_tail:MAC:length"]
    assert length == pytest.approx(4.161, abs=1e-3)
    vt_x0 = problem["data:geometry:vertical_tail:MAC:at25percent:x:local"]
    assert vt_x0 == pytest.approx(2.321, abs=1e-3)
    vt_z0 = problem["data:geometry:vertical_tail:MAC:z"]
    assert vt_z0 == pytest.approx(2.716, abs=1e-3)


def test_compute_vt_chords(input_xml):
    """Tests computation of the vertical tail chords"""

    input_list = [
        "data:geometry:vertical_tail:aspect_ratio",
        "data:geometry:vertical_tail:area",
        "data:geometry:vertical_tail:taper_ratio",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    component = ComputeVTChords()

    problem = run_system(component, input_vars)

    span = problem["data:geometry:vertical_tail:span"]
    assert span == pytest.approx(6.62, abs=1e-2)
    root_chord = problem["data:geometry:vertical_tail:root:chord"]
    assert root_chord == pytest.approx(5.837, abs=1e-3)
    tip_chord = problem["data:geometry:vertical_tail:tip:chord"]
    assert tip_chord == pytest.approx(1.751, abs=1e-3)


def test_compute_vt_sweep(input_xml):
    """Tests computation of the vertical tail sweep"""

    input_list = [
        "data:geometry:vertical_tail:root:chord",
        "data:geometry:vertical_tail:tip:chord",
        "data:geometry:vertical_tail:span",
        "data:geometry:vertical_tail:sweep_25",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    component = ComputeVTSweep()

    problem = run_system(component, input_vars)

    sweep_0 = problem["data:geometry:vertical_tail:sweep_0"]
    assert sweep_0 == pytest.approx(40.515, abs=1e-3)
    sweep_100 = problem["data:geometry:vertical_tail:sweep_100"]
    assert sweep_100 == pytest.approx(13.35, abs=1e-2)


def test_compute_vt_distance(input_xml):
    """Tests computation of the vertical tail distance"""

    input_list = [
        "data:geometry:fuselage:length",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:has_T_tail",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()
    problem = run_system(ComputeVTDistance(), input_vars)

    lp_vt = problem["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
    assert lp_vt == pytest.approx(16.55, abs=1e-2)


def test_compute_vt_cl(input_xml):
    """Tests computation of the vertical tail lift coefficient"""

    input_list = [
        "data:TLAR:cruise_mach",
        "data:geometry:vertical_tail:aspect_ratio",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:has_T_tail",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()
    problem = run_system(ComputeVTClalpha(), input_vars)

    cl_alpha = problem["data:aerodynamics:vertical_tail:cruise:CL_alpha"]
    assert cl_alpha == pytest.approx(2.55, abs=1e-2)


def test_compute_vt_local_positions(input_xml):
    """Tests computation of the vertical tail local positions"""

    input_list = [
        "data:geometry:vertical_tail:taper_ratio",
        "data:geometry:vertical_tail:MAC:at25percent:x:local",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:span",
        "data:geometry:vertical_tail:sweep_25",
    ]
    input_vars = input_xml.read(only=input_list).to_ivc()
    problem = run_system(ComputeVTLocalPositions(), input_vars)

    tip_le_x_local = problem["data:geometry:vertical_tail:tip:leading_edge:x:local"]
    assert tip_le_x_local == pytest.approx(4.20, abs=1e-2)
    root_le_x_local = problem["data:geometry:vertical_tail:root:leading_edge:x:local"]
    assert root_le_x_local == pytest.approx(-0.65, abs=1e-2)
    mac_le_x_local = problem["data:geometry:vertical_tail:MAC:leading_edge:x:local"]
    assert mac_le_x_local == pytest.approx(1.34, abs=1e-2)


def test_compute_vt_global_positions(input_xml):
    """Tests computation of the vertical tail global positions"""

    input_list = [
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:vertical_tail:MAC:leading_edge:x:local",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:tip:leading_edge:x:local",
    ]
    input_vars = input_xml.read(only=input_list).to_ivc()
    problem = run_system(ComputeChordGlobalPositions(), input_vars)

    tip_le_x = problem["data:geometry:vertical_tail:tip:leading_edge:x"]
    assert tip_le_x == pytest.approx(34.78, abs=1e-2)
    root_le_x = problem["data:geometry:vertical_tail:root:leading_edge:x"]
    assert root_le_x == pytest.approx(30.58, abs=1e-2)
    mac_le_x = problem["data:geometry:vertical_tail:MAC:leading_edge:x"]
    assert mac_le_x == pytest.approx(31.92, abs=1e-2)


def test_geometry_total_area(input_xml):
    """Tests computation of the total area"""

    input_list = [
        "data:geometry:wing:wetted_area",
        "data:geometry:fuselage:wetted_area",
        "data:geometry:horizontal_tail:wetted_area",
        "data:geometry:vertical_tail:wetted_area",
        "data:geometry:propulsion:nacelle:wetted_area",
        "data:geometry:propulsion:pylon:wetted_area",
        "data:geometry:propulsion:engine:count",
    ]

    input_vars = input_xml.read(only=input_list).to_ivc()

    component = ComputeWettedArea()

    problem = run_system(component, input_vars)

    total_surface = problem["data:geometry:aircraft:wetted_area"]
    assert total_surface == pytest.approx(783.997, abs=1e-3)
