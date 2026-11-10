"""
Estimation of wing weight
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from scipy.constants import g

from .constants import SERVICE_WING_MASS


@oad.RegisterSubmodel(SERVICE_WING_MASS, "fastoad.submodel.weight.mass.airframe.wing.legacy")
class WingWeight(om.ExplicitComponent):
    """
    Wing weight estimation

    This is done by summing following estimations:

    - mass from sizing to flexion forces
    - mass from sizing to shear forces
    - mass of ribs
    - mass of reinforcements
    - mass of secondary parts

    Based on :cite:`supaero:2014`, mass contribution A1
    """

    def setup(self):
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:wing:kink:thickness_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:outer_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan, units="unitless")
        self.add_input("data:geometry:propulsion:layout", val=np.nan, units="unitless")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:cs25:sizing_load", val=np.nan, units="N")
        self.add_input("tuning:weight:airframe:wing:mass:k", val=1.0, units="unitless")
        self.add_input("tuning:weight:airframe:wing:mass:offset", val=0.0, units="kg")
        self.add_input(
            "tuning:weight:airframe:wing:bending_sizing:mass:k", val=1.0, units="unitless"
        )
        self.add_input(
            "tuning:weight:airframe:wing:bending_sizing:mass:offset", val=0.0, units="kg"
        )
        self.add_input("tuning:weight:airframe:wing:shear_sizing:mass:k", val=1.0, units="unitless")
        self.add_input("tuning:weight:airframe:wing:shear_sizing:mass:offset", val=0.0, units="kg")
        self.add_input("tuning:weight:airframe:wing:ribs:mass:k", val=1.0, units="unitless")
        self.add_input("tuning:weight:airframe:wing:ribs:mass:offset", val=0.0, units="kg")
        self.add_input(
            "tuning:weight:airframe:wing:reinforcements:mass:k", val=1.0, units="unitless"
        )
        self.add_input(
            "tuning:weight:airframe:wing:reinforcements:mass:offset", val=0.0, units="kg"
        )
        self.add_input(
            "tuning:weight:airframe:wing:secondary_parts:mass:k", val=1.0, units="unitless"
        )
        self.add_input(
            "tuning:weight:airframe:wing:secondary_parts:mass:offset", val=0.0, units="kg"
        )
        self.add_input("settings:weight:airframe:wing:mass:k_mvo", val=1.39, units="unitless")

        self.add_output("data:weight:airframe:wing:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        toc_root = inputs["data:geometry:wing:root:thickness_ratio"]
        toc_kink = inputs["data:geometry:wing:kink:thickness_ratio"]
        toc_tip = inputs["data:geometry:wing:tip:thickness_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        cantilevered_area = inputs["data:geometry:wing:outer_area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        mlw = inputs["data:weight:aircraft:MLW"]
        max_nm = inputs["data:mission:sizing:cs25:sizing_load"] / g
        engine_count = inputs["data:geometry:propulsion:engine:count"]
        propulsion_layout = np.round(inputs["data:geometry:propulsion:layout"])
        engine_on_fuselage = propulsion_layout == 2
        if engine_on_fuselage:
            k_voil = 1.1
        elif engine_count >= 4:
            k_voil = 1.0
        else:
            k_voil = 1.05

        # K factors
        k_a1 = inputs["tuning:weight:airframe:wing:mass:k"]
        offset_a1 = inputs["tuning:weight:airframe:wing:mass:offset"]
        k_a11 = inputs["tuning:weight:airframe:wing:bending_sizing:mass:k"]
        offset_a11 = inputs["tuning:weight:airframe:wing:bending_sizing:mass:offset"]
        k_a12 = inputs["tuning:weight:airframe:wing:shear_sizing:mass:k"]
        offset_a12 = inputs["tuning:weight:airframe:wing:shear_sizing:mass:offset"]
        k_a13 = inputs["tuning:weight:airframe:wing:ribs:mass:k"]
        offset_a13 = inputs["tuning:weight:airframe:wing:ribs:mass:offset"]
        k_a14 = inputs["tuning:weight:airframe:wing:reinforcements:mass:k"]
        offset_a14 = inputs["tuning:weight:airframe:wing:reinforcements:mass:offset"]
        k_a15 = inputs["tuning:weight:airframe:wing:secondary_parts:mass:k"]
        offset_a15 = inputs["tuning:weight:airframe:wing:secondary_parts:mass:offset"]
        k_mvo = inputs["settings:weight:airframe:wing:mass:k_mvo"]

        toc_mean = (3 * toc_root + 2 * toc_kink + toc_tip) / 6

        # A11=Mass of the wing due to flexion
        temp_a11 = (
            5.922e-5
            * k_voil
            * ((max_nm / (l2_wing * toc_mean)) * (span / np.cos(sweep_25)) ** 2.0) ** 0.9
        )
        weight_a11 = k_a11 * temp_a11 + offset_a11

        # A12=Mass of the wing due to shear
        temp_a12 = 5.184e-4 * k_voil * (max_nm * span / np.cos(sweep_25)) ** 0.9
        weight_a12 = k_a12 * temp_a12 + offset_a12

        # A13=Mass of the wing due to the ribs
        temp_a13 = k_voil * (1.7009 * wing_area + 10 ** (-3) * max_nm)
        weight_a13 = k_a13 * temp_a13 + offset_a13

        # A14=Mass of the wing due to reinforcements
        temp_a14 = 4.4e-3 * k_voil * mlw**1.0169
        weight_a14 = k_a14 * temp_a14 + offset_a14

        # A15=Mass of the wing due to secondary parts
        temp_a15 = 0.3285 * k_voil * mtow**0.35 * cantilevered_area * k_mvo
        weight_a15 = k_a15 * temp_a15 + offset_a15

        temp_a1 = weight_a11 + weight_a12 + weight_a13 + weight_a14 + weight_a15

        outputs["data:weight:airframe:wing:mass"] = k_a1 * temp_a1 + offset_a1

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        toc_root = inputs["data:geometry:wing:root:thickness_ratio"]
        toc_kink = inputs["data:geometry:wing:kink:thickness_ratio"]
        toc_tip = inputs["data:geometry:wing:tip:thickness_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        cantilevered_area = inputs["data:geometry:wing:outer_area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        mlw = inputs["data:weight:aircraft:MLW"]
        max_nm = inputs["data:mission:sizing:cs25:sizing_load"] / g
        engine_count = inputs["data:geometry:propulsion:engine:count"]
        propulsion_layout = np.round(inputs["data:geometry:propulsion:layout"])
        engine_on_fuselage = propulsion_layout == 2
        if engine_on_fuselage:
            k_voil = 1.1
        elif engine_count >= 4:
            k_voil = 1.0
        else:
            k_voil = 1.05

        k_a1 = inputs["tuning:weight:airframe:wing:mass:k"]
        k_a11 = inputs["tuning:weight:airframe:wing:bending_sizing:mass:k"]
        k_a12 = inputs["tuning:weight:airframe:wing:shear_sizing:mass:k"]
        k_a13 = inputs["tuning:weight:airframe:wing:ribs:mass:k"]
        k_a14 = inputs["tuning:weight:airframe:wing:reinforcements:mass:k"]
        k_a15 = inputs["tuning:weight:airframe:wing:secondary_parts:mass:k"]
        k_mvo = inputs["settings:weight:airframe:wing:mass:k_mvo"]

        toc_mean = (3 * toc_root + 2 * toc_kink + toc_tip) / 6
        cos_sweep = np.cos(sweep_25)

        # A11 = mass of the wing due to flexion
        a11_factor = (max_nm / (l2_wing * toc_mean)) * (span / cos_sweep) ** 2.0
        temp_a11 = 5.922e-5 * k_voil * a11_factor**0.9
        d_temp_a11_d_max_nm = 0.9 * temp_a11 / max_nm
        d_temp_a11_d_l2 = -0.9 * temp_a11 / l2_wing
        d_temp_a11_d_toc_mean = -0.9 * temp_a11 / toc_mean
        d_temp_a11_d_span = 1.8 * temp_a11 / span
        d_temp_a11_d_sweep = 1.8 * temp_a11 * np.tan(sweep_25)

        # A12 = mass of the wing due to shear
        a12_factor = max_nm * span / cos_sweep
        temp_a12 = 5.184e-4 * k_voil * a12_factor**0.9
        d_temp_a12_d_max_nm = 0.9 * temp_a12 / max_nm
        d_temp_a12_d_span = 0.9 * temp_a12 / span
        d_temp_a12_d_sweep = 0.9 * temp_a12 * np.tan(sweep_25)

        # A13 = mass of the wing due to the ribs
        temp_a13 = k_voil * (1.7009 * wing_area + 1e-3 * max_nm)
        d_temp_a13_d_max_nm = k_voil * 1e-3

        # A14 = mass of the wing due to reinforcements
        temp_a14 = 4.4e-3 * k_voil * mlw**1.0169
        d_temp_a14_d_mlw = 4.4e-3 * k_voil * 1.0169 * mlw**0.0169

        # A15 = mass of the wing due to secondary parts
        temp_a15 = 0.3285 * k_voil * mtow**0.35 * cantilevered_area * k_mvo
        d_temp_a15_d_mtow = 0.3285 * k_voil * 0.35 * mtow ** (-0.65) * cantilevered_area * k_mvo
        d_temp_a15_d_area = 0.3285 * k_voil * mtow**0.35 * k_mvo
        d_temp_a15_d_k_mvo = 0.3285 * k_voil * mtow**0.35 * cantilevered_area

        # Tuning factors are linear multipliers, so their partials are the corresponding
        # un-tuned sub-masses.
        offset_a11 = inputs["tuning:weight:airframe:wing:bending_sizing:mass:offset"]
        offset_a12 = inputs["tuning:weight:airframe:wing:shear_sizing:mass:offset"]
        offset_a13 = inputs["tuning:weight:airframe:wing:ribs:mass:offset"]
        offset_a14 = inputs["tuning:weight:airframe:wing:reinforcements:mass:offset"]
        offset_a15 = inputs["tuning:weight:airframe:wing:secondary_parts:mass:offset"]
        temp_a1 = (
            k_a11 * temp_a11
            + offset_a11
            + k_a12 * temp_a12
            + offset_a12
            + k_a13 * temp_a13
            + offset_a13
            + k_a14 * temp_a14
            + offset_a14
            + k_a15 * temp_a15
            + offset_a15
        )

        partials["data:weight:airframe:wing:mass", "tuning:weight:airframe:wing:mass:k"] = temp_a1
        partials["data:weight:airframe:wing:mass", "tuning:weight:airframe:wing:mass:offset"] = 1.0

        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:bending_sizing:mass:k",
        ] = k_a1 * temp_a11
        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:bending_sizing:mass:offset",
        ] = k_a1
        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:shear_sizing:mass:k",
        ] = k_a1 * temp_a12
        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:shear_sizing:mass:offset",
        ] = k_a1
        partials["data:weight:airframe:wing:mass", "tuning:weight:airframe:wing:ribs:mass:k"] = (
            k_a1 * temp_a13
        )
        partials[
            "data:weight:airframe:wing:mass", "tuning:weight:airframe:wing:ribs:mass:offset"
        ] = k_a1
        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:reinforcements:mass:k",
        ] = k_a1 * temp_a14
        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:reinforcements:mass:offset",
        ] = k_a1
        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:secondary_parts:mass:k",
        ] = k_a1 * temp_a15
        partials[
            "data:weight:airframe:wing:mass",
            "tuning:weight:airframe:wing:secondary_parts:mass:offset",
        ] = k_a1

        d_temp_a1_d_max_nm = (
            k_a11 * d_temp_a11_d_max_nm + k_a12 * d_temp_a12_d_max_nm + k_a13 * d_temp_a13_d_max_nm
        )
        partials["data:weight:airframe:wing:mass", "data:mission:sizing:cs25:sizing_load"] = (
            k_a1 * d_temp_a1_d_max_nm / g
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:root:thickness_ratio"] = (
            k_a1 * k_a11 * d_temp_a11_d_toc_mean * 3.0 / 6.0
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:kink:thickness_ratio"] = (
            k_a1 * k_a11 * d_temp_a11_d_toc_mean * 2.0 / 6.0
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:tip:thickness_ratio"] = (
            k_a1 * k_a11 * d_temp_a11_d_toc_mean / 6.0
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:root:chord"] = (
            k_a1 * k_a11 * d_temp_a11_d_l2
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:span"] = k_a1 * (
            k_a11 * d_temp_a11_d_span + k_a12 * d_temp_a12_d_span
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:sweep_25"] = k_a1 * (
            k_a11 * d_temp_a11_d_sweep + k_a12 * d_temp_a12_d_sweep
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:area"] = (
            k_a1 * k_a13 * k_voil * 1.7009
        )
        partials["data:weight:airframe:wing:mass", "data:weight:aircraft:MLW"] = (
            k_a1 * k_a14 * d_temp_a14_d_mlw
        )
        partials["data:weight:airframe:wing:mass", "data:weight:aircraft:MTOW"] = (
            k_a1 * k_a15 * d_temp_a15_d_mtow
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:outer_area"] = (
            k_a1 * k_a15 * d_temp_a15_d_area
        )
        partials["data:weight:airframe:wing:mass", "settings:weight:airframe:wing:mass:k_mvo"] = (
            k_a1 * k_a15 * d_temp_a15_d_k_mvo
        )
