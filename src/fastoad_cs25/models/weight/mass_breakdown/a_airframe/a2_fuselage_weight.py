"""
Estimation of fuselage weight
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

from .constants import SERVICE_FUSELAGE_MASS


@oad.RegisterSubmodel(
    SERVICE_FUSELAGE_MASS, "fastoad.submodel.weight.mass.airframe.fuselage.legacy"
)
class FuselageWeight(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on a statistical analysis. See :cite:`supaero:2014`, mass contribution A2
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:mission:sizing:cs25:envelope:max_sizing_load_1", val=np.nan, units="N")
        self.add_input("tuning:weight:airframe:fuselage:mass:k", val=1.0, units="unitless")
        self.add_input("tuning:weight:airframe:fuselage:mass:offset", val=0.0, units="kg")
        self.add_input("settings:weight:airframe:fuselage:mass:k_lg", val=1.05, units="unitless")
        self.add_input("settings:weight:airframe:fuselage:mass:k_fus", val=1.0, units="unitless")

        self.add_output("data:weight:airframe:fuselage:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuselage_wet_area = inputs["data:geometry:fuselage:wetted_area"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        n1m1 = inputs["data:mission:sizing:cs25:envelope:max_sizing_load_1"]
        k_a2 = inputs["tuning:weight:airframe:fuselage:mass:k"]
        offset_a2 = inputs["tuning:weight:airframe:fuselage:mass:offset"]
        k_lg = inputs["settings:weight:airframe:fuselage:mass:k_lg"]
        k_fus = inputs["settings:weight:airframe:fuselage:mass:k_fus"]

        temp_a2 = (
            fuselage_wet_area
            * (10 + 1.2 * np.sqrt(width_max * height_max) + 0.00019 * n1m1 / g / height_max**1.7)
            * k_lg
            * k_fus
        )
        outputs["data:weight:airframe:fuselage:mass"] = k_a2 * temp_a2 + offset_a2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuselage_wet_area = inputs["data:geometry:fuselage:wetted_area"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        n1m1 = inputs["data:mission:sizing:cs25:envelope:max_sizing_load_1"]
        k_a2 = inputs["tuning:weight:airframe:fuselage:mass:k"]
        k_lg = inputs["settings:weight:airframe:fuselage:mass:k_lg"]
        k_fus = inputs["settings:weight:airframe:fuselage:mass:k_fus"]

        fuselage_diameter = np.sqrt(width_max * height_max)
        bracket = 10 + 1.2 * fuselage_diameter + 0.00019 * n1m1 / g / height_max**1.7
        d_bracket_d_width = 0.6 * height_max / fuselage_diameter
        d_bracket_d_height = (
            0.6 * width_max / fuselage_diameter - 0.00019 * n1m1 / g * 1.7 * height_max ** (-2.7)
        )
        d_bracket_d_n1m1 = 0.00019 / g / height_max**1.7

        common = k_a2 * fuselage_wet_area * k_lg * k_fus
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:wetted_area"] = (
            k_a2 * bracket * k_lg * k_fus
        )
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"] = (
            common * d_bracket_d_width
        )
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"] = (
            common * d_bracket_d_height
        )
        partials[
            "data:weight:airframe:fuselage:mass",
            "data:mission:sizing:cs25:envelope:max_sizing_load_1",
        ] = common * d_bracket_d_n1m1
        partials[
            "data:weight:airframe:fuselage:mass",
            "settings:weight:airframe:fuselage:mass:k_lg",
        ] = k_a2 * fuselage_wet_area * bracket * k_fus
        partials[
            "data:weight:airframe:fuselage:mass",
            "settings:weight:airframe:fuselage:mass:k_fus",
        ] = k_a2 * fuselage_wet_area * bracket * k_lg
        partials["data:weight:airframe:fuselage:mass", "tuning:weight:airframe:fuselage:mass:k"] = (
            fuselage_wet_area * bracket * k_lg * k_fus
        )
        partials[
            "data:weight:airframe:fuselage:mass", "tuning:weight:airframe:fuselage:mass:offset"
        ] = 1.0
