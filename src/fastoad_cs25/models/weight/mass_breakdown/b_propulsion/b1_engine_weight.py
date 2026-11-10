"""
Estimation of engine weight
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

from .constants import SERVICE_ENGINE_MASS


# TODO:  this is also provided by class RubberEngine
@oad.RegisterSubmodel(SERVICE_ENGINE_MASS, "fastoad.submodel.weight.mass.propulsion.engine.legacy")
class EngineWeight(om.ExplicitComponent):
    """
    Engine weight estimation

    Uses model described in :cite:`roux:2005`, p.74
    """

    def setup(self):
        self.add_input("data:propulsion:MTO_thrust", val=np.nan, units="N")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan, units="unitless")
        self.add_input("tuning:weight:propulsion:engine:mass:k", val=1.0, units="unitless")
        self.add_input("tuning:weight:propulsion:engine:mass:offset", val=0.0, units="kg")

        self.add_output("data:weight:propulsion:engine:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sea_level_thrust = inputs["data:propulsion:MTO_thrust"]
        n_engines = inputs["data:geometry:propulsion:engine:count"]
        k_b1 = inputs["tuning:weight:propulsion:engine:mass:k"]
        offset_b1 = inputs["tuning:weight:propulsion:engine:mass:offset"]

        if sea_level_thrust < 80000:
            temp_b1 = 22.2e-3 * sea_level_thrust
        else:
            temp_b1 = 14.1e-3 * sea_level_thrust + 648

        temp_b1 *= n_engines * 1.55
        outputs["data:weight:propulsion:engine:mass"] = k_b1 * temp_b1 + offset_b1

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sea_level_thrust = inputs["data:propulsion:MTO_thrust"]
        n_engines = inputs["data:geometry:propulsion:engine:count"]
        k_b1 = inputs["tuning:weight:propulsion:engine:mass:k"]

        if sea_level_thrust < 80000:
            base_thrust = 22.2e-3 * sea_level_thrust
            d_base_d_thrust = 22.2e-3
        else:
            base_thrust = 14.1e-3 * sea_level_thrust + 648
            d_base_d_thrust = 14.1e-3

        temp_b1 = base_thrust * n_engines * 1.55

        partials["data:weight:propulsion:engine:mass", "data:propulsion:MTO_thrust"] = (
            k_b1 * d_base_d_thrust * n_engines * 1.55
        )
        partials["data:weight:propulsion:engine:mass", "data:geometry:propulsion:engine:count"] = (
            k_b1 * base_thrust * 1.55
        )
        partials["data:weight:propulsion:engine:mass", "tuning:weight:propulsion:engine:mass:k"] = (
            temp_b1
        )
        partials[
            "data:weight:propulsion:engine:mass",
            "tuning:weight:propulsion:engine:mass:offset",
        ] = 1.0
