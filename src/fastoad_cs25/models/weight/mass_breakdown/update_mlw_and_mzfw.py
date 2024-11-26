"""
Main component for mass breakdown
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2025 ONERA & ISAE-SUPAERO
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
from openmdao.core.explicitcomponent import ExplicitComponent

from .constants import SERVICE_MLW_MZFW


@oad.RegisterSubmodel(SERVICE_MLW_MZFW, "fastoad.submodel.weight.mass.mlw_mzfw.legacy")
class UpdateMLWandMZFW(ExplicitComponent):
    """
    Computes Maximum Landing Weight and Maximum Zero Fuel Weight from
    Overall Empty Weight and Maximum Payload, using an empirical law.
    """

    def setup(self):
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")
        # Add tuning factor for mzfw and mlw
        self.add_input("tuning:weight:aircraft:mlw_mzfw_ratio", val=1.06)

        self.add_output("data:weight:aircraft:MZFW", units="kg")
        self.add_output("data:weight:aircraft:MLW", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        owe = inputs["data:weight:aircraft:OWE"]
        max_pl = inputs["data:weight:aircraft:max_payload"]
        mlw_mzfw_ratio = inputs["tuning:weight:aircraft:mlw_mzfw_ratio"]

        mzfw = owe + max_pl
        mlw = mlw_mzfw_ratio * mzfw

        outputs["data:weight:aircraft:MZFW"] = mzfw
        outputs["data:weight:aircraft:MLW"] = mlw
