"""
test module for modules in aerodynamics/components
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

import os.path as pth

import numpy as np
from fastoad.io import VariableIO
from fastoad.testing import run_system
from numpy.testing import assert_allclose
from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from pytest import approx
from stdatm import Atmosphere

from ...constants import PolarType
from ..cd0 import CD0
from ..cd_compressibility import CdCompressibility
from ..cd_trim import CdTrim
from ..compute_alpha import ComputeAlpha
from ..compute_cl_alpha import ComputeCLAlpha
from ..compute_polar import ComputePolar
from ..compute_reynolds import ComputeReynolds
from ..high_lift_aero import ComputeDeltaHighLift
from ..oswald import InducedDragCoefficient, OswaldCoefficient


def get_indep_var_comp(var_names):
    """Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "aerodynamics_inputs.xml"))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()
    return ivc


def test_high_lift_aero():
    """Tests ComputeDeltaHighLift"""
    input_list = [
        "data:geometry:wing:sweep_0",
        "data:geometry:wing:sweep_100_outer",
        "data:geometry:flap:chord_ratio",
        "data:geometry:flap:span_ratio",
        "data:geometry:slat:chord_ratio",
        "data:geometry:slat:span_ratio",
    ]

    def get_cl_cd(slat_angle, flap_angle, mach, landing_flag, multi_slot_flag):
        ivc = get_indep_var_comp(input_list)
        if landing_flag:
            ivc.add_output("data:mission:sizing:landing:slat_angle", slat_angle, units="deg")
            ivc.add_output("data:mission:sizing:landing:flap_angle", flap_angle, units="deg")
            ivc.add_output("data:aerodynamics:aircraft:landing:mach", mach, units="unitless")
        else:
            ivc.add_output("data:mission:sizing:takeoff:slat_angle", slat_angle, units="deg")
            ivc.add_output("data:mission:sizing:takeoff:flap_angle", flap_angle, units="deg")
            ivc.add_output("data:aerodynamics:aircraft:takeoff:mach", mach, units="unitless")
        if multi_slot_flag:
            ivc.add_output(
                "tuning:aerodynamics:high_lift_devices:landing:CL:multi_slotted_flap_effect:k",
                1.1,
                units="unitless",
            )
            ivc.add_output(
                "tuning:aerodynamics:high_lift_devices:landing:CD:multi_slotted_flap_effect:k",
                1.01,
                units="unitless",
            )
        component = ComputeDeltaHighLift()
        component.options["landing_flag"] = landing_flag
        problem = run_system(component, ivc)
        if landing_flag:
            return (
                problem["data:aerodynamics:high_lift_devices:landing:CL"],
                problem["data:aerodynamics:high_lift_devices:landing:CD"],
            )

        return (
            problem["data:aerodynamics:high_lift_devices:takeoff:CL"],
            problem["data:aerodynamics:high_lift_devices:takeoff:CD"],
        )

    cl, cd = get_cl_cd(18, 10, 0.2, False, False)
    assert cl == approx(0.516, abs=1e-3)
    assert cd == approx(0.01430, abs=1e-5)

    cl, cd = get_cl_cd(27, 35, 0.4, False, False)
    assert cl == approx(1.431, abs=1e-3)
    assert cd == approx(0.04644, abs=1e-5)

    cl, cd = get_cl_cd(27, 35, 0.4, False, True)
    assert cl == approx(1.564, abs=1e-3)
    assert cd == approx(0.04675, abs=1e-5)

    cl, cd = get_cl_cd(22, 20, 0.2, True, False)
    assert cl == approx(0.935, abs=1e-3)
    assert cd == approx(0.02220, abs=1e-5)

    cl, cd = get_cl_cd(22, 20, 0.2, True, True)
    assert cl == approx(1.021, abs=1e-3)
    assert cd == approx(0.02230, abs=1e-5)


def test_oswald_coefficient():
    """Tests OswaldCoefficient"""
    input_list = [
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:sweep_25",
    ]

    def get_coeff(mach, low_speed_aero=False):
        ivc = get_indep_var_comp(input_list)
        if low_speed_aero:
            ivc.add_output("data:aerodynamics:aircraft:takeoff:mach", mach, units="unitless")
        else:
            ivc.add_output("data:TLAR:cruise_mach", mach, units="unitless")
        problem = run_system(OswaldCoefficient(low_speed_aero=low_speed_aero), ivc)
        if low_speed_aero:
            return problem["data:aerodynamics:aircraft:low_speed:oswald_coefficient"]
        else:
            return problem["data:aerodynamics:aircraft:cruise:oswald_coefficient"]

    assert get_coeff(0.2, True) == approx(0.8490, abs=1e-4)
    # FIXME: Obtained Oswald coeff is too low, especially for high speed.
    assert get_coeff(0.8, False) == approx(0.7451, abs=1e-4)


def test_induced_drag_coefficient():
    """Tests InducedDragCoefficient"""
    input_list = [
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:aerodynamics:aircraft:low_speed:oswald_coefficient",
        "data:aerodynamics:aircraft:cruise:oswald_coefficient",
    ]

    def get_coeff(mach, low_speed_aero=False):
        ivc = get_indep_var_comp(input_list)
        if low_speed_aero:
            ivc.add_output(
                "data:aerodynamics:aircraft:low_speed:oswald_coefficient", mach, units="unitless"
            )
        else:
            ivc.add_output(
                "data:aerodynamics:aircraft:cruise:oswald_coefficient", mach, units="unitless"
            )
        problem = run_system(InducedDragCoefficient(low_speed_aero=low_speed_aero), ivc)
        if low_speed_aero:
            return problem["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
        else:
            return problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]

    assert get_coeff(0.8490, True) == approx(0.0465, abs=1e-4)
    assert get_coeff(0.7451, False) == approx(0.0530, abs=1e-4)


def test_cd0():
    """Tests CD0 (test of the group for comparison to legacy)"""
    input_list = [
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:wetted_area",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:thickness_ratio",
        "data:geometry:horizontal_tail:wetted_area",
        "data:geometry:wing:area",
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:fan:length",
        "data:geometry:propulsion:nacelle:length",
        "data:geometry:propulsion:nacelle:wetted_area",
        "data:geometry:propulsion:pylon:length",
        "data:geometry:propulsion:pylon:wetted_area",
        "data:geometry:aircraft:wetted_area",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:vertical_tail:thickness_ratio",
        "data:geometry:vertical_tail:wetted_area",
        "data:geometry:wing:area",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:thickness_ratio",
        "data:geometry:wing:wetted_area",
    ]

    def get_cd0(alt, mach, cl, low_speed_aero):
        atm = Atmosphere(alt)
        atm.mach = mach
        reynolds = atm.unitary_reynolds

        ivc = get_indep_var_comp(input_list)
        if low_speed_aero:
            ivc.add_output("data:aerodynamics:aircraft:takeoff:mach", mach, units="unitless")
            ivc.add_output("data:aerodynamics:wing:low_speed:reynolds", reynolds, units="unitless")
            ivc.add_output(
                "data:aerodynamics:aircraft:low_speed:CL", 150 * [cl], units="unitless"
            )  # needed because size of input array is fixed
        else:
            ivc.add_output("data:TLAR:cruise_mach", mach, units="unitless")
            ivc.add_output("data:aerodynamics:wing:cruise:reynolds", reynolds, units="unitless")
            ivc.add_output(
                "data:aerodynamics:aircraft:cruise:CL", 150 * [cl], units="unitless"
            )  # needed because size of input array is fixed
        problem = run_system(CD0(low_speed_aero=low_speed_aero), ivc)
        # We test here only the first element of the polar as all the values are constants
        if low_speed_aero:
            return problem.get_val("data:aerodynamics:aircraft:low_speed:CD0", units="unitless")[0]
        else:
            return problem.get_val("data:aerodynamics:aircraft:cruise:CD0", units="unitless")[0]

    assert get_cd0(35000, 0.78, 0.5, False) == approx(0.01975, abs=1e-5)
    assert get_cd0(0, 0.2, 0.9, True) == approx(0.02727, abs=1e-5)


def test_cd_compressibility():
    """Tests CdCompressibility"""

    def get_cd_compressibility(mach, cl, sweep, thickness_ratio, delta_charac_mach=0.0):
        ivc = IndepVarComp()
        ivc.add_output(
            "data:aerodynamics:aircraft:cruise:CL", 150 * [cl]
        )  # needed because size of input array is fixed
        ivc.add_output("data:TLAR:cruise_mach", mach, units="unitless")
        ivc.add_output("data:geometry:wing:sweep_25", sweep, units="deg")
        ivc.add_output("data:geometry:wing:thickness_ratio", thickness_ratio, units="unitless")
        ivc.add_output(
            "tuning:aerodynamics:aircraft:cruise:CD:compressibility:characteristic_mach_increment",
            delta_charac_mach,
            units="unitless",
        )

        problem = run_system(CdCompressibility(), ivc)
        # We test here only the first element of the polar as all the values are constants
        return problem.get_val(
            "data:aerodynamics:aircraft:cruise:CD:compressibility", units="unitless"
        )[0]

    assert get_cd_compressibility(0.2, 0.9, 28, 0.12) == approx(0.0, abs=1e-5)
    assert get_cd_compressibility(0.78, 0.2, 28, 0.12) == approx(0.00028, abs=1e-5)
    assert get_cd_compressibility(0.78, 0.35, 28, 0.12) == approx(0.00028, abs=1e-5)
    assert get_cd_compressibility(0.78, 0.5, 28, 0.12) == approx(0.00045, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.2, 28, 0.12) == approx(0.00359, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.35, 28, 0.12) == approx(0.00359, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.5, 28, 0.12) == approx(0.00580, abs=1e-5)

    assert get_cd_compressibility(0.84, 0.2, 35, 0.12) == approx(0.00023, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.35, 35, 0.12) == approx(0.00023, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.5, 35, 0.12) == approx(0.00039, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.2, 28, 0.12, 0.05) == approx(0.00043, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.35, 28, 0.12, 0.05) == approx(0.00043, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.5, 28, 0.12, 0.05) == approx(0.00069, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.5, 28, 0.12) == approx(0.00579, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.8, 28, 0.12) == approx(0.26769, abs=1e-5)
    assert get_cd_compressibility(0.84, 1.0, 28, 0.12) == approx(0.5, abs=1e-5)  # capped

    assert get_cd_compressibility(0.84, 0.2, 28, 0.10) == approx(0.00137, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.35, 28, 0.10) == approx(0.00137, abs=1e-5)
    assert get_cd_compressibility(0.84, 0.5, 28, 0.10) == approx(0.00221, abs=1e-5)


def test_cd_trim():
    """Tests CdTrim"""

    def get_cd_trim(cl):
        ivc = IndepVarComp()
        ivc.add_output(
            "data:aerodynamics:aircraft:cruise:CL", 150 * [cl], units="unitless"
        )  # needed because size of input array is fixed
        problem = run_system(CdTrim(), ivc)
        # We test here only the first element of the polar as all the values are constants
        return problem.get_val("data:aerodynamics:aircraft:cruise:CD:trim", units="unitless")[0]

    assert get_cd_trim(0.5) == approx(0.0002945, abs=1e-6)
    assert get_cd_trim(0.9) == approx(0.0005301, abs=1e-6)


def test_polar_high_speed():
    """Tests ComputePolar"""

    # Need to plug Cd modules, Reynolds and Oswald

    input_list = [
        "tuning:aerodynamics:aircraft:cruise:CD:k",
        "tuning:aerodynamics:aircraft:cruise:CD:offset",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:wetted_area",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:area",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:thickness_ratio",
        "data:geometry:horizontal_tail:wetted_area",
        "data:geometry:wing:area",
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:fan:length",
        "data:geometry:propulsion:nacelle:length",
        "data:geometry:propulsion:nacelle:wetted_area",
        "data:geometry:propulsion:pylon:length",
        "data:geometry:propulsion:pylon:wetted_area",
        "data:geometry:wing:area",
        "data:geometry:aircraft:wetted_area",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:vertical_tail:thickness_ratio",
        "data:geometry:vertical_tail:wetted_area",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:thickness_ratio",
        "data:geometry:wing:wetted_area",
        "data:geometry:wing:span",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:TLAR:cruise_mach",
        "data:mission:sizing:main_route:cruise:altitude",
    ]
    group = Group()
    group.add_subsystem("reynolds", ComputeReynolds(), promotes=["*"])
    group.add_subsystem("oswald", OswaldCoefficient(), promotes=["*"])
    group.add_subsystem("induced_drag_coeff", InducedDragCoefficient(), promotes=["*"])
    group.add_subsystem("cd0", CD0(), promotes=["*"])
    group.add_subsystem("cd_compressibility", CdCompressibility(), promotes=["*"])
    group.add_subsystem("cd_trim", CdTrim(), promotes=["*"])
    group.add_subsystem("polar", ComputePolar(), promotes=["*"])

    ivc = get_indep_var_comp(input_list)
    ivc.add_output(
        "data:aerodynamics:aircraft:cruise:CL", np.arange(0.0, 1.5, 0.01), units="unitless"
    )

    problem = run_system(group, ivc)

    cd = problem["data:aerodynamics:aircraft:cruise:CD"]
    cl = problem["data:aerodynamics:aircraft:cruise:CL"]

    assert cd[cl == 0.0] == approx(0.02102, abs=1e-5)
    assert cd[cl == 0.2] == approx(0.02281, abs=1e-5)
    assert cd[cl == 0.42] == approx(0.02977, abs=1e-5)
    assert cd[cl == 0.85] == approx(0.24041, abs=1e-5)

    assert problem["data:aerodynamics:aircraft:cruise:optimal_CL"] == approx(0.51, abs=1e-3)
    assert problem["data:aerodynamics:aircraft:cruise:optimal_CD"] == approx(0.03487, abs=1e-5)


def test_polar_low_speed():
    """Tests ComputePolar"""

    # Need to plug Cd modules, Reynolds and Oswald

    input_list = [
        "data:aerodynamics:aircraft:takeoff:mach",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:thickness_ratio",
        "data:geometry:wing:wetted_area",
        "data:geometry:wing:MAC:length",
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:wetted_area",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:thickness_ratio",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:wetted_area",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:thickness_ratio",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:vertical_tail:wetted_area",
        "data:geometry:propulsion:pylon:length",
        "data:geometry:propulsion:nacelle:length",
        "data:geometry:propulsion:pylon:wetted_area",
        "data:geometry:propulsion:nacelle:wetted_area",
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:fan:length",
        "data:geometry:aircraft:wetted_area",
        "tuning:aerodynamics:aircraft:cruise:CD:k",
        "tuning:aerodynamics:aircraft:cruise:CD:offset",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset",
    ]
    group = Group()
    group.add_subsystem("reynolds", ComputeReynolds(low_speed_aero=True), promotes=["*"])
    group.add_subsystem("oswald", OswaldCoefficient(low_speed_aero=True), promotes=["*"])
    group.add_subsystem(
        "induced_drag_coeff", InducedDragCoefficient(low_speed_aero=True), promotes=["*"]
    )
    group.add_subsystem("cd0", CD0(low_speed_aero=True), promotes=["*"])
    group.add_subsystem("cd_trim", CdTrim(low_speed_aero=True), promotes=["*"])
    group.add_subsystem("polar", ComputePolar(polar_type=PolarType.LOW_SPEED), promotes=["*"])

    ivc = get_indep_var_comp(input_list)
    ivc.add_output(
        "data:aerodynamics:aircraft:low_speed:CL", np.arange(0.0, 1.5, 0.01), units="unitless"
    )

    problem = run_system(group, ivc)

    cd = problem["data:aerodynamics:aircraft:low_speed:CD"]
    cl = problem["data:aerodynamics:aircraft:low_speed:CL"]

    assert cd[cl == 0.5] == approx(0.033441, abs=1e-5)
    assert cd[cl == 1.0] == approx(0.077523, abs=1e-5)


def test_polar_high_lift():
    """Tests ComputePolar"""

    # Need to plug Cd modules, Reynolds and Oswald

    input_list = [
        "data:aerodynamics:aircraft:takeoff:mach",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:thickness_ratio",
        "data:geometry:wing:wetted_area",
        "data:geometry:wing:MAC:length",
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:wetted_area",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:thickness_ratio",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:wetted_area",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:thickness_ratio",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:vertical_tail:wetted_area",
        "data:geometry:propulsion:pylon:length",
        "data:geometry:propulsion:nacelle:length",
        "data:geometry:propulsion:pylon:wetted_area",
        "data:geometry:propulsion:nacelle:wetted_area",
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:fan:length",
        "data:geometry:aircraft:wetted_area",
        "tuning:aerodynamics:aircraft:cruise:CD:k",
        "tuning:aerodynamics:aircraft:cruise:CD:offset",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset",
        "data:aerodynamics:high_lift_devices:takeoff:CL",
        "data:aerodynamics:high_lift_devices:takeoff:CD",
    ]
    group = Group()
    group.add_subsystem("reynolds", ComputeReynolds(low_speed_aero=True), promotes=["*"])
    group.add_subsystem("oswald", OswaldCoefficient(low_speed_aero=True), promotes=["*"])
    group.add_subsystem(
        "induced_drag_coeff", InducedDragCoefficient(low_speed_aero=True), promotes=["*"]
    )
    group.add_subsystem("cd0", CD0(low_speed_aero=True), promotes=["*"])
    group.add_subsystem("cd_trim", CdTrim(low_speed_aero=True), promotes=["*"])
    group.add_subsystem("polar", ComputePolar(polar_type=PolarType.TAKEOFF), promotes=["*"])

    ivc = get_indep_var_comp(input_list)
    ivc.add_output(
        "data:aerodynamics:aircraft:low_speed:CL", np.arange(0.0, 1.5, 0.01), units="unitless"
    )

    problem = run_system(group, ivc)

    cd = problem["data:aerodynamics:aircraft:takeoff:CD"]
    cl = problem["data:aerodynamics:aircraft:takeoff:CL"]

    assert cd[cl == 1.006] == approx([0.083137], abs=1e-5)
    assert cd[cl == 1.506] == approx([0.150439], abs=1e-5)


def test_compute_cl_alpha_low_speed():
    """Tests group ComputeAerodynamicsLowSpeed"""
    input_list = [
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:wing:span",
        "data:geometry:wing:aspect_ratio",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:area",
        "data:geometry:wing:tip:thickness_ratio",
    ]
    ivc = get_indep_var_comp(input_list)

    problem = run_system(ComputeCLAlpha(low_speed_aero=True), ivc)

    assert problem["data:aerodynamics:aircraft:low_speed:CL_alpha"] == approx(5.0, abs=1e-1)


def test_compute_cl_alpha_cruise():
    """Tests computation of the wing lift coefficient"""
    input_list = [
        "data:TLAR:cruise_mach",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:wing:span",
        "data:geometry:wing:aspect_ratio",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:area",
        "data:geometry:wing:tip:thickness_ratio",
    ]
    ivc = get_indep_var_comp(input_list)
    problem = run_system(ComputeCLAlpha(), ivc)
    cl_alpha = problem["data:aerodynamics:aircraft:cruise:CL_alpha"]
    assert cl_alpha == approx(6.49, abs=1e-2)


def test_compute_alpha_low_speed():
    """Tests computation of the wing lift coefficient"""
    input_list = [
        "data:aerodynamics:aircraft:low_speed:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        # Using default CL0 value of 0.2
    ]
    ivc = get_indep_var_comp(input_list)

    problem = run_system(ComputeAlpha(low_speed_aero=True), ivc)
    alpha = problem.get_val("data:aerodynamics:aircraft:low_speed:AoA", units="deg")

    assert_allclose(
        alpha,
        np.linspace(-2.29183118, 14.78231111, 150),
        atol=1e-4,
    )

    problem.set_val("data:aerodynamics:aircraft:low_speed:CL0", 0.15)
    problem.run_model()
    alpha = problem.get_val("data:aerodynamics:aircraft:low_speed:AoA", units="deg")
    assert_allclose(
        alpha,
        np.linspace(-2.29183118, 14.78231111, 150) + np.degrees(0.05 / 5.0),
        atol=1e-4,
    )


def test_compute_alpha_cruise():
    """Tests computation of the wing lift coefficient"""
    input_list = [
        "data:aerodynamics:aircraft:cruise:CL",
        "data:aerodynamics:aircraft:cruise:CL_alpha",
        # Using default CL0 value of 0.1
    ]
    ivc = get_indep_var_comp(input_list)

    problem = run_system(ComputeAlpha(), ivc)
    alpha = problem.get_val("data:aerodynamics:aircraft:cruise:AoA", units="deg")
    assert_allclose(
        alpha,
        np.linspace(-0.882832, 7.9454856, 101),
        atol=1e-4,
    )

    problem.set_val("data:aerodynamics:aircraft:cruise:CL0", 0.05)
    problem.run_model()
    alpha = problem.get_val("data:aerodynamics:aircraft:cruise:AoA", units="deg")
    assert_allclose(
        alpha,
        np.linspace(-0.882832, 7.9454856, 101) + np.degrees(0.05 / 6.49),
        atol=1e-4,
    )
