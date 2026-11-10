"""
Estimation of pylons weight
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

from .constants import SERVICE_PYLONS_MASS


@oad.RegisterSubmodel(SERVICE_PYLONS_MASS, "fastoad.submodel.weight.mass.airframe.pylons.legacy")
class PylonsWeight(om.ExplicitComponent):
    """
    Weight estimation for pylons

    Based on formula in :cite:`supaero:2014`, mass contribution A6
    """

    def setup(self):
        self.add_input("data:geometry:propulsion:pylon:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:layout", val=np.nan, units="unitless")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan, units="unitless")
        self.add_input("tuning:weight:airframe:pylon:mass:k", val=1.0, units="unitless")
        self.add_input("tuning:weight:airframe:pylon:mass:offset", val=0.0, units="kg")

        self.add_output("data:weight:airframe:pylon:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wet_area_pylon = inputs["data:geometry:propulsion:pylon:wetted_area"]
        weight_engine = inputs["data:weight:propulsion:engine:mass"]
        n_engines = inputs["data:geometry:propulsion:engine:count"]
        k_a6 = inputs["tuning:weight:airframe:pylon:mass:k"]
        offset_a6 = inputs["tuning:weight:airframe:pylon:mass:offset"]
        propulsion_layout = np.round(inputs["data:geometry:propulsion:layout"])

        if propulsion_layout == 1.0:
            temp_a6 = (
                1.2
                * wet_area_pylon**0.5
                * n_engines
                * (23 + 0.588 * (weight_engine / n_engines) ** 0.708)
            )
        else:
            temp_a6 = 0.08 * weight_engine

        outputs["data:weight:airframe:pylon:mass"] = k_a6 * temp_a6 + offset_a6

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wet_area_pylon = inputs["data:geometry:propulsion:pylon:wetted_area"]
        weight_engine = inputs["data:weight:propulsion:engine:mass"]
        n_engines = inputs["data:geometry:propulsion:engine:count"]
        k_a6 = inputs["tuning:weight:airframe:pylon:mass:k"]
        propulsion_layout = np.round(inputs["data:geometry:propulsion:layout"])

        output_name = "data:weight:airframe:pylon:mass"
        if propulsion_layout == 1.0:
            ratio_pow = (weight_engine / n_engines) ** 0.708
            bracket = 23 + 0.588 * ratio_pow
            temp_a6 = 1.2 * np.sqrt(wet_area_pylon) * n_engines * bracket

            partials[output_name, "data:geometry:propulsion:pylon:wetted_area"] = (
                k_a6 * 0.6 / np.sqrt(wet_area_pylon) * n_engines * bracket
            )
            partials[output_name, "data:weight:propulsion:engine:mass"] = (
                k_a6
                * 1.2
                * np.sqrt(wet_area_pylon)
                * 0.588
                * 0.708
                * ratio_pow
                * n_engines
                / weight_engine
            )
            partials[output_name, "data:geometry:propulsion:engine:count"] = (
                k_a6 * 1.2 * np.sqrt(wet_area_pylon) * (23 + 0.588 * 0.292 * ratio_pow)
            )
        else:
            temp_a6 = 0.08 * weight_engine
            partials[output_name, "data:weight:propulsion:engine:mass"] = k_a6 * 0.08

        partials[output_name, "tuning:weight:airframe:pylon:mass:k"] = temp_a6
        partials[output_name, "tuning:weight:airframe:pylon:mass:offset"] = 1.0
