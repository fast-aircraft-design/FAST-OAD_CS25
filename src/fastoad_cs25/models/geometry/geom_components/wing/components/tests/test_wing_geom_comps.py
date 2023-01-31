"""
Test module for geometry functions of cg components
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2023 ONERA & ISAE-SUPAERO
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
from fastoad._utils.testing import run_system

from ..compute_b_50 import ComputeB50
from ..compute_l1_l4 import ComputeL1AndL4Wing
from ..compute_l2_l3 import ComputeL2AndL3Wing
from ..compute_mac_wing import ComputeMACWing
from ..compute_mfw import ComputeMFW
from ..compute_sweep_wing import ComputeSweepWing
from ..compute_toc_wing import ComputeToCWing
from ..compute_wet_area_wing import ComputeWetAreaWing
from ..compute_x_wing import ComputeXWing
from ..compute_y_wing import ComputeYWing


def test_geometry_wing_b50():
    """Tests computation of the wing B50"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:span", 31.603, units="m")
    input_vars.add_output("data:geometry:wing:root:virtual_chord", 4.953, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 7.222, units="m")

    problem = run_system(ComputeB50(), input_vars)
    wing_b_50 = problem["data:geometry:wing:b_50"]
    assert wing_b_50 == pytest.approx(34.166, abs=1e-3)


def test_geometry_wing_l1_l4():
    """Tests computation of the wing chords (l1 and l4)"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:span", 31.603, units="m")
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units=None)
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")

    problem = run_system(ComputeL1AndL4Wing(), input_vars)

    wing_l1 = problem["data:geometry:wing:root:virtual_chord"]
    assert wing_l1 == pytest.approx(4.953, abs=1e-3)
    wing_l4 = problem["data:geometry:wing:tip:chord"]
    assert wing_l4 == pytest.approx(1.882, abs=1e-3)


def test_geometry_wing_l2_l3():
    """Tests computation of the wing chords (l2 and l3)"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:span", 31.603, units="m")
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units=None)
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:root:virtual_chord", 4.953, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")

    problem = run_system(ComputeL2AndL3Wing(), input_vars)
    wing_l2 = problem["data:geometry:wing:root:chord"]
    assert wing_l2 == pytest.approx(6.26, abs=1e-2)
    wing_l3 = problem["data:geometry:wing:kink:chord"]
    assert wing_l3 == pytest.approx(3.985, abs=1e-3)
    taper_ratio = problem["data:geometry:wing:taper_ratio"]
    assert taper_ratio == pytest.approx(0.301, abs=1e-3)

    # With no kink
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:span", 31.603, units="m")
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units=None)
    input_vars.add_output("data:geometry:wing:kink:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:root:virtual_chord", 4.953, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")

    problem = run_system(ComputeL2AndL3Wing(), input_vars)
    wing_l2 = problem["data:geometry:wing:root:chord"]
    assert wing_l2 == pytest.approx(4.953, abs=1e-2)
    wing_l3 = problem["data:geometry:wing:kink:chord"]
    assert wing_l3 == pytest.approx(4.953, abs=1e-3)
    taper_ratio = problem["data:geometry:wing:taper_ratio"]
    assert taper_ratio == pytest.approx(0.38, abs=1e-3)


def test_geometry_wing_mac():
    """Tests computation of the wing mean aerodynamic chord"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 2.275, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 7.222, units="m")

    problem = run_system(ComputeMACWing(), input_vars)
    wing_l0 = problem["data:geometry:wing:MAC:length"]
    assert wing_l0 == pytest.approx(4.457, abs=1e-3)
    wing_x0 = problem["data:geometry:wing:MAC:leading_edge:x:local"]
    assert wing_x0 == pytest.approx(2.361, abs=1e-3)
    wing_y0 = problem["data:geometry:wing:MAC:y"]
    assert wing_y0 == pytest.approx(6.293, abs=1e-3)


def test_geometry_wing_mfw():
    """Tests computation of the wing max fuel weight"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units=None)
    input_vars.add_output("data:geometry:wing:root:thickness_ratio", 0.159, units=None)
    input_vars.add_output("data:geometry:wing:tip:thickness_ratio", 0.11, units=None)

    problem = run_system(ComputeMFW(), input_vars)
    mfw = problem["data:weight:aircraft:MFW"]
    assert mfw == pytest.approx(19284.7, abs=1e-1)


def test_geometry_wing_sweep():
    """Tests computation of the wing sweeps"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:kink:leading_edge:x:local", 2.275, units="m")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")
    input_vars.add_output("data:geometry:wing:tip:leading_edge:x:local", 7.222, units="m")

    problem = run_system(ComputeSweepWing(), input_vars)
    sweep_0 = problem["data:geometry:wing:sweep_0"]
    assert sweep_0 == pytest.approx(27.55, abs=1e-2)
    sweep_100_inner = problem["data:geometry:wing:sweep_100_inner"]
    assert sweep_100_inner == pytest.approx(0.0, abs=1e-1)
    sweep_100_outer = problem["data:geometry:wing:sweep_100_outer"]
    assert sweep_100_outer == pytest.approx(16.7, abs=1e-1)


def test_geometry_wing_toc():
    """Tests computation of the wing ToC (Thickness of Chord)"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units=None)
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.4, units=None)

    problem = run_system(ComputeToCWing(), input_vars)
    toc_aero = problem["data:geometry:wing:thickness_ratio"]
    assert toc_aero == pytest.approx(0.128, abs=1e-3)
    toc_root = problem["data:geometry:wing:root:thickness_ratio"]
    assert toc_root == pytest.approx(0.159, abs=1e-3)
    toc_kink = problem["data:geometry:wing:kink:thickness_ratio"]
    assert toc_kink == pytest.approx(0.121, abs=1e-3)
    toc_tip = problem["data:geometry:wing:tip:thickness_ratio"]
    assert toc_tip == pytest.approx(0.11, abs=1e-2)

    # With no kink
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units=None)
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.0, units=None)

    problem = run_system(ComputeToCWing(), input_vars)
    toc_aero = problem["data:geometry:wing:thickness_ratio"]
    assert toc_aero == pytest.approx(0.128, abs=1e-3)
    toc_root = problem["data:geometry:wing:root:thickness_ratio"]
    assert toc_root == pytest.approx(0.159, abs=1e-3)
    toc_kink = problem["data:geometry:wing:kink:thickness_ratio"]
    assert toc_kink == pytest.approx(0.159, abs=1e-3)
    toc_tip = problem["data:geometry:wing:tip:thickness_ratio"]
    assert toc_tip == pytest.approx(0.11, abs=1e-2)


def test_geometry_wing_wet_area():
    """Tests computation of the wing wet area"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:root:chord", 6.26, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")

    problem = run_system(ComputeWetAreaWing(), input_vars)
    area_pf = problem["data:geometry:wing:outer_area"]
    assert area_pf == pytest.approx(100.303, abs=1e-3)
    wet_area = problem["data:geometry:wing:wetted_area"]
    assert wet_area == pytest.approx(200.607, abs=1e-3)


def test_geometry_wing_x():
    """Tests computation of the wing Xs"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:kink:chord", 3.985, units="m")
    input_vars.add_output("data:geometry:wing:kink:y", 6.321, units="m")
    input_vars.add_output("data:geometry:wing:root:virtual_chord", 4.953, units="m")
    input_vars.add_output("data:geometry:wing:root:y", 1.96, units="m")
    input_vars.add_output("data:geometry:wing:tip:chord", 1.882, units="m")
    input_vars.add_output("data:geometry:wing:tip:y", 15.801, units="m")

    problem = run_system(ComputeXWing(), input_vars)
    wing_x3 = problem["data:geometry:wing:kink:leading_edge:x:local"]
    assert wing_x3 == pytest.approx(2.275, abs=1e-3)
    wing_x4 = problem["data:geometry:wing:tip:leading_edge:x:local"]
    assert wing_x4 == pytest.approx(7.222, abs=1e-3)


def test_geometry_wing_y():
    """Tests computation of the wing Ys"""

    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units=None)
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.4, units=None)

    problem = run_system(ComputeYWing(), input_vars)
    span = problem["data:geometry:wing:span"]
    assert span == pytest.approx(34.4, abs=1e-1)
    wing_y2 = problem["data:geometry:wing:root:y"]
    assert wing_y2 == pytest.approx(1.96, abs=1e-2)
    wing_y3 = problem["data:geometry:wing:kink:y"]
    assert wing_y3 == pytest.approx(6.88, abs=1e-2)
    wing_y4 = problem["data:geometry:wing:tip:y"]
    assert wing_y4 == pytest.approx(17.2, abs=1e-1)

    # With no kink
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units=None)
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.0, units=None)

    problem = run_system(ComputeYWing(), input_vars)
    span = problem["data:geometry:wing:span"]
    assert span == pytest.approx(34.4, abs=1e-1)
    wing_y2 = problem["data:geometry:wing:root:y"]
    assert wing_y2 == pytest.approx(1.96, abs=1e-2)
    wing_y3 = problem["data:geometry:wing:kink:y"]
    assert wing_y3 == pytest.approx(1.96, abs=1e-2)
    wing_y4 = problem["data:geometry:wing:tip:y"]
    assert wing_y4 == pytest.approx(17.2, abs=1e-1)
