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

from ..compute_wing import ComputeWingGeometry


def test_compute_wing_with_kink():
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units="unitless")
    input_vars.add_output("data:geometry:fuselage:maximum_height", 4.06, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units="unitless")
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.4, units="unitless")
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units="unitless")

    problem = run_system(ComputeWingGeometry(), input_vars)

    assert problem.get_val(
        "data:geometry:wing:MAC:leading_edge:x:local", units="m"
    ) == pytest.approx(2.524, abs=1e-3)
    assert problem.get_val("data:geometry:wing:MAC:length", units="m") == pytest.approx(
        4.182, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:MAC:y", units="m") == pytest.approx(6.710, abs=1e-3)
    assert problem.get_val("data:geometry:wing:b_50", units="m") == pytest.approx(37.33, abs=1e-3)
    assert problem.get_val("data:geometry:wing:kink:chord", units="m") == pytest.approx(
        3.540, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:kink:leading_edge:x:local", units="m"
    ) == pytest.approx(2.516, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:kink:thickness_ratio", units="unitless"
    ) == pytest.approx(0.121, abs=1e-3)
    assert problem.get_val("data:geometry:wing:kink:y", units="m") == pytest.approx(6.88, abs=1e-2)
    assert problem.get_val("data:geometry:wing:outer_area", units="m**2") == pytest.approx(
        101.105, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:chord", units="m") == pytest.approx(
        6.056, abs=1e-2
    )
    assert problem.get_val(
        "data:geometry:wing:root:thickness_ratio", units="unitless"
    ) == pytest.approx(0.159, abs=1e-3)
    assert problem.get_val("data:geometry:wing:root:virtual_chord", units="m") == pytest.approx(
        4.426, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:y", units="m") == pytest.approx(1.96, abs=1e-2)
    assert problem.get_val("data:geometry:wing:span", units="m") == pytest.approx(34.4, abs=1e-1)
    assert problem.get_val("data:geometry:wing:sweep_0", units="deg") == pytest.approx(
        27.08, abs=1e-2
    )
    assert problem.get_val("data:geometry:wing:sweep_100_inner", units="deg") == pytest.approx(
        0.0, abs=1e-1
    )
    assert problem.get_val("data:geometry:wing:sweep_100_outer", units="deg") == pytest.approx(
        18.33, abs=1e-1
    )
    assert problem.get_val("data:geometry:wing:taper_ratio", units="unitless") == pytest.approx(
        0.2777, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:thickness_ratio", units="unitless") == pytest.approx(
        0.128, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:tip:chord", units="m") == pytest.approx(
        1.682, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:tip:leading_edge:x:local", units="m"
    ) == pytest.approx(7.793, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:tip:thickness_ratio", units="unitless"
    ) == pytest.approx(0.110, abs=1e-2)
    assert problem.get_val("data:geometry:wing:tip:y", units="m") == pytest.approx(17.2, abs=1e-1)
    assert problem.get_val("data:geometry:wing:wetted_area", units="m**2") == pytest.approx(
        202.209, abs=1e-3
    )
    assert problem.get_val("data:weight:aircraft:MFW", units="kg") == pytest.approx(
        19322.5, abs=1e-1
    )


def test_compute_wing_with_absolute_kink():
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units="unitless")
    input_vars.add_output("data:geometry:fuselage:maximum_height", 4.06, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units="unitless")
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units="unitless")
    input_vars.add_output("data:geometry:wing:kink:y", 6.88, units="m")

    problem = run_system(ComputeWingGeometry(impose_absolute_kink=True), input_vars)

    assert problem.get_val(
        "data:geometry:wing:MAC:leading_edge:x:local", units="m"
    ) == pytest.approx(2.524, abs=1e-3)
    assert problem.get_val("data:geometry:wing:MAC:length", units="m") == pytest.approx(
        4.182, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:MAC:y", units="m") == pytest.approx(6.710, abs=1e-3)
    assert problem.get_val("data:geometry:wing:b_50", units="m") == pytest.approx(37.33, abs=1e-3)
    assert problem.get_val("data:geometry:wing:kink:chord", units="m") == pytest.approx(
        3.540, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:kink:leading_edge:x:local", units="m"
    ) == pytest.approx(2.516, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:kink:thickness_ratio", units="unitless"
    ) == pytest.approx(0.121, abs=1e-3)
    assert problem.get_val("data:geometry:wing:outer_area", units="m**2") == pytest.approx(
        101.105, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:chord", units="m") == pytest.approx(
        6.056, abs=1e-2
    )
    assert problem.get_val(
        "data:geometry:wing:root:thickness_ratio", units="unitless"
    ) == pytest.approx(0.159, abs=1e-3)
    assert problem.get_val("data:geometry:wing:root:virtual_chord", units="m") == pytest.approx(
        4.426, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:y", units="m") == pytest.approx(1.96, abs=1e-2)
    assert problem.get_val("data:geometry:wing:span", units="m") == pytest.approx(34.4, abs=1e-1)
    assert problem.get_val("data:geometry:wing:sweep_0", units="deg") == pytest.approx(
        27.08, abs=1e-2
    )
    assert problem.get_val("data:geometry:wing:sweep_100_inner", units="deg") == pytest.approx(
        0.0, abs=1e-1
    )
    assert problem.get_val("data:geometry:wing:sweep_100_outer", units="deg") == pytest.approx(
        18.33, abs=1e-1
    )
    assert problem.get_val("data:geometry:wing:taper_ratio", units="unitless") == pytest.approx(
        0.2777, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:thickness_ratio", units="unitless") == pytest.approx(
        0.128, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:tip:chord", units="m") == pytest.approx(
        1.682, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:tip:leading_edge:x:local", units="m"
    ) == pytest.approx(7.793, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:tip:thickness_ratio", units="unitless"
    ) == pytest.approx(0.110, abs=1e-2)
    assert problem.get_val("data:geometry:wing:tip:y", units="m") == pytest.approx(17.2, abs=1e-1)
    assert problem.get_val("data:geometry:wing:wetted_area", units="m**2") == pytest.approx(
        202.209, abs=1e-3
    )
    assert problem.get_val("data:weight:aircraft:MFW", units="kg") == pytest.approx(
        19322.5, abs=1e-1
    )


def test_compute_wing_without_kink():
    # Method 1 : kink span_ratio = 0.
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units="unitless")
    input_vars.add_output("data:geometry:fuselage:maximum_height", 4.06, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units="unitless")
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.0, units="unitless")
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units="unitless")

    problem = run_system(ComputeWingGeometry(), input_vars)

    assert problem.get_val(
        "data:geometry:wing:MAC:leading_edge:x:local", units="m"
    ) == pytest.approx(2.825, abs=1e-3)
    assert problem.get_val("data:geometry:wing:MAC:length", units="m") == pytest.approx(
        3.892, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:MAC:y", units="m") == pytest.approx(7.268, abs=1e-3)
    assert problem.get_val("data:geometry:wing:b_50", units="m") == pytest.approx(37.253, abs=1e-3)
    assert problem.get_val("data:geometry:wing:kink:chord", units="m") == pytest.approx(
        5.003, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:kink:leading_edge:x:local", units="m"
    ) == pytest.approx(0.0, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:kink:thickness_ratio", units="unitless"
    ) == pytest.approx(0.159, abs=1e-3)
    assert problem.get_val("data:geometry:wing:kink:y", units="m") == pytest.approx(1.96, abs=1e-2)
    assert problem.get_val("data:geometry:wing:outer_area", units="m**2") == pytest.approx(
        105.231, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:chord", units="m") == pytest.approx(
        5.003, abs=1e-2
    )
    assert problem.get_val(
        "data:geometry:wing:root:thickness_ratio", units="unitless"
    ) == pytest.approx(0.159, abs=1e-3)
    assert problem.get_val("data:geometry:wing:root:virtual_chord", units="m") == pytest.approx(
        5.003, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:y", units="m") == pytest.approx(1.96, abs=1e-2)
    assert problem.get_val("data:geometry:wing:span", units="m") == pytest.approx(34.4, abs=1e-1)
    assert problem.get_val("data:geometry:wing:sweep_0", units="deg") == pytest.approx(
        27.35, abs=1e-2
    )
    assert problem.get_val("data:geometry:wing:sweep_100_outer", units="deg") == pytest.approx(
        17.41, abs=1e-1
    )
    assert problem.get_val("data:geometry:wing:taper_ratio", units="unitless") == pytest.approx(
        0.38, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:thickness_ratio", units="unitless") == pytest.approx(
        0.128, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:tip:chord", units="m") == pytest.approx(
        1.901, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:tip:leading_edge:x:local", units="m"
    ) == pytest.approx(7.883, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:tip:thickness_ratio", units="unitless"
    ) == pytest.approx(0.110, abs=1e-2)
    assert problem.get_val("data:geometry:wing:tip:y", units="m") == pytest.approx(17.2, abs=1e-1)
    assert problem.get_val("data:geometry:wing:wetted_area", units="m**2") == pytest.approx(
        210.461, abs=1e-3
    )
    assert problem.get_val("data:weight:aircraft:MFW", units="kg") == pytest.approx(
        19322.5, abs=1e-1
    )

    # Method 2 : sweep_100_ratio = 1.0
    input_vars = om.IndepVarComp()
    input_vars.add_output("data:TLAR:cruise_mach", 0.78, units="unitless")
    input_vars.add_output("data:geometry:fuselage:maximum_height", 4.06, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_width", 3.92, units="m")
    input_vars.add_output("data:geometry:wing:area", 124.843, units="m**2")
    input_vars.add_output("data:geometry:wing:aspect_ratio", 9.48, units="unitless")
    input_vars.add_output("data:geometry:wing:kink:span_ratio", 0.4, units="unitless")
    input_vars.add_output("data:geometry:wing:sweep_25", 25.0, units="deg")
    input_vars.add_output("data:geometry:wing:sweep_100_ratio", 1.0, units="unitless")
    input_vars.add_output("data:geometry:wing:virtual_taper_ratio", 0.38, units="unitless")

    problem = run_system(ComputeWingGeometry(), input_vars)

    assert problem.get_val(
        "data:geometry:wing:MAC:leading_edge:x:local", units="m"
    ) == pytest.approx(2.825, abs=1e-3)
    assert problem.get_val("data:geometry:wing:MAC:length", units="m") == pytest.approx(
        3.892, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:MAC:y", units="m") == pytest.approx(7.268, abs=1e-3)
    assert problem.get_val("data:geometry:wing:b_50", units="m") == pytest.approx(37.253, abs=1e-3)
    assert problem.get_val("data:geometry:wing:kink:chord", units="m") == pytest.approx(
        4.002, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:kink:leading_edge:x:local", units="m"
    ) == pytest.approx(2.545, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:kink:thickness_ratio", units="unitless"
    ) == pytest.approx(0.121, abs=1e-3)
    assert problem.get_val("data:geometry:wing:kink:y", units="m") == pytest.approx(6.88, abs=1e-2)
    assert problem.get_val("data:geometry:wing:outer_area", units="m**2") == pytest.approx(
        105.231, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:chord", units="m") == pytest.approx(
        5.003, abs=1e-2
    )
    assert problem.get_val(
        "data:geometry:wing:root:thickness_ratio", units="unitless"
    ) == pytest.approx(0.159, abs=1e-3)
    assert problem.get_val("data:geometry:wing:root:virtual_chord", units="m") == pytest.approx(
        5.003, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:root:y", units="m") == pytest.approx(1.96, abs=1e-2)
    assert problem.get_val("data:geometry:wing:span", units="m") == pytest.approx(34.4, abs=1e-1)
    assert problem.get_val("data:geometry:wing:sweep_0", units="deg") == pytest.approx(
        27.35, abs=1e-2
    )
    assert problem.get_val("data:geometry:wing:sweep_100_outer", units="deg") == pytest.approx(
        17.41, abs=1e-1
    )
    assert problem.get_val("data:geometry:wing:taper_ratio", units="unitless") == pytest.approx(
        0.38, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:thickness_ratio", units="unitless") == pytest.approx(
        0.128, abs=1e-3
    )
    assert problem.get_val("data:geometry:wing:tip:chord", units="m") == pytest.approx(
        1.901, abs=1e-3
    )
    assert problem.get_val(
        "data:geometry:wing:tip:leading_edge:x:local", units="m"
    ) == pytest.approx(7.883, abs=1e-3)
    assert problem.get_val(
        "data:geometry:wing:tip:thickness_ratio", units="unitless"
    ) == pytest.approx(0.110, abs=1e-2)
    assert problem.get_val("data:geometry:wing:tip:y", units="m") == pytest.approx(17.2, abs=1e-1)
    assert problem.get_val("data:geometry:wing:wetted_area", units="m**2") == pytest.approx(
        210.461, abs=1e-3
    )
    assert problem.get_val("data:weight:aircraft:MFW", units="kg") == pytest.approx(
        19322.5, abs=1e-1
    )
