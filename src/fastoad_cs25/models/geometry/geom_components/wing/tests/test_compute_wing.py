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

import openmdao.api as om
import pytest
from fastoad._utils.testing import run_system

from ..compute_wing import ComputeWingGeometry


def test_compute_wing_with_kink():
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units=None)
    input_vars.add_output("data:geometry:fuselage:maximum_height", 4.06, units=None)
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units=None)
    input_vars.add_output("data:geometry:wing:area", 124.843, units=None)
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units=None)
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.4, units=None)
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units=None)
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units=None)

    problem = run_system(ComputeWingGeometry(), input_vars)

    assert problem["data:aerodynamics:aircraft:cruise:CL_alpha"] == pytest.approx(6.418, abs=1e-2)
    assert problem["data:geometry:wing:MAC:leading_edge:x:local"] == pytest.approx(2.524, abs=1e-3)
    assert problem["data:geometry:wing:MAC:length"] == pytest.approx(4.182, abs=1e-3)
    assert problem["data:geometry:wing:MAC:y"] == pytest.approx(6.710, abs=1e-3)
    assert problem["data:geometry:wing:b_50"] == pytest.approx(37.33, abs=1e-3)
    assert problem["data:geometry:wing:kink:chord"] == pytest.approx(3.540, abs=1e-3)
    assert problem["data:geometry:wing:kink:leading_edge:x:local"] == pytest.approx(2.516, abs=1e-3)
    assert problem["data:geometry:wing:kink:thickness_ratio"] == pytest.approx(0.121, abs=1e-3)
    assert problem["data:geometry:wing:kink:y"] == pytest.approx(6.88, abs=1e-2)
    assert problem["data:geometry:wing:outer_area"] == pytest.approx(101.105, abs=1e-3)
    assert problem["data:geometry:wing:root:chord"] == pytest.approx(6.056, abs=1e-2)
    assert problem["data:geometry:wing:root:thickness_ratio"] == pytest.approx(0.159, abs=1e-3)
    assert problem["data:geometry:wing:root:virtual_chord"] == pytest.approx(4.426, abs=1e-3)
    assert problem["data:geometry:wing:root:y"] == pytest.approx(1.96, abs=1e-2)
    assert problem["data:geometry:wing:span"] == pytest.approx(34.4, abs=1e-1)
    assert problem["data:geometry:wing:sweep_0"] == pytest.approx(27.08, abs=1e-2)
    assert problem["data:geometry:wing:sweep_100_inner"] == pytest.approx(0.0, abs=1e-1)
    assert problem["data:geometry:wing:sweep_100_outer"] == pytest.approx(18.33, abs=1e-1)
    assert problem["data:geometry:wing:taper_ratio"] == pytest.approx(0.2777, abs=1e-3)
    assert problem["data:geometry:wing:thickness_ratio"] == pytest.approx(0.128, abs=1e-3)
    assert problem["data:geometry:wing:tip:chord"] == pytest.approx(1.682, abs=1e-3)
    assert problem["data:geometry:wing:tip:leading_edge:x:local"] == pytest.approx(7.793, abs=1e-3)
    assert problem["data:geometry:wing:tip:thickness_ratio"] == pytest.approx(0.110, abs=1e-2)
    assert problem["data:geometry:wing:tip:y"] == pytest.approx(17.2, abs=1e-1)
    assert problem["data:geometry:wing:wetted_area"] == pytest.approx(202.209, abs=1e-3)
    assert problem["data:weight:aircraft:MFW"] == pytest.approx(19322.5, abs=1e-1)


def test_compute_wing_without_kink():
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units=None)
    input_vars.add_output("data:geometry:fuselage:maximum_height", 4.06, units=None)
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units=None)
    input_vars.add_output("data:geometry:wing:area", 124.843, units=None)
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units=None)
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.0, units=None)
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units=None)
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units=None)

    problem = run_system(ComputeWingGeometry(), input_vars)

    assert problem["data:aerodynamics:aircraft:cruise:CL_alpha"] == pytest.approx(6.682, abs=1e-2)
    assert problem["data:geometry:wing:MAC:leading_edge:x:local"] == pytest.approx(2.825, abs=1e-3)
    assert problem["data:geometry:wing:MAC:length"] == pytest.approx(3.892, abs=1e-3)
    assert problem["data:geometry:wing:MAC:y"] == pytest.approx(7.268, abs=1e-3)
    assert problem["data:geometry:wing:b_50"] == pytest.approx(37.253, abs=1e-3)
    assert problem["data:geometry:wing:kink:chord"] == pytest.approx(5.003, abs=1e-3)
    assert problem["data:geometry:wing:kink:leading_edge:x:local"] == pytest.approx(0.0, abs=1e-3)
    assert problem["data:geometry:wing:kink:thickness_ratio"] == pytest.approx(0.159, abs=1e-3)
    assert problem["data:geometry:wing:kink:y"] == pytest.approx(1.96, abs=1e-2)
    assert problem["data:geometry:wing:outer_area"] == pytest.approx(105.231, abs=1e-3)
    assert problem["data:geometry:wing:root:chord"] == pytest.approx(5.003, abs=1e-2)
    assert problem["data:geometry:wing:root:thickness_ratio"] == pytest.approx(0.159, abs=1e-3)
    assert problem["data:geometry:wing:root:virtual_chord"] == pytest.approx(5.003, abs=1e-3)
    assert problem["data:geometry:wing:root:y"] == pytest.approx(1.96, abs=1e-2)
    assert problem["data:geometry:wing:span"] == pytest.approx(34.4, abs=1e-1)
    assert problem["data:geometry:wing:sweep_0"] == pytest.approx(27.35, abs=1e-2)
    assert problem["data:geometry:wing:sweep_100_inner"] == pytest.approx(17.41, abs=1e-1)
    assert problem["data:geometry:wing:sweep_100_outer"] == pytest.approx(17.41, abs=1e-1)
    assert problem["data:geometry:wing:taper_ratio"] == pytest.approx(0.38, abs=1e-3)
    assert problem["data:geometry:wing:thickness_ratio"] == pytest.approx(0.128, abs=1e-3)
    assert problem["data:geometry:wing:tip:chord"] == pytest.approx(1.901, abs=1e-3)
    assert problem["data:geometry:wing:tip:leading_edge:x:local"] == pytest.approx(7.883, abs=1e-3)
    assert problem["data:geometry:wing:tip:thickness_ratio"] == pytest.approx(0.110, abs=1e-2)
    assert problem["data:geometry:wing:tip:y"] == pytest.approx(17.2, abs=1e-1)
    assert problem["data:geometry:wing:wetted_area"] == pytest.approx(210.461, abs=1e-3)
    assert problem["data:weight:aircraft:MFW"] == pytest.approx(19322.5, abs=1e-1)
