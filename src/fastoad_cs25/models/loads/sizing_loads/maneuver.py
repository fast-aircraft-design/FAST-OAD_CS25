"""
Python module for the computation of maneuver sizing load cases.
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from openmdao.api import convert_units

from ..constants import SERVICE_MANEUVER_LOADS


@oad.RegisterSubmodel(SERVICE_MANEUVER_LOADS, "fastoad.submodel.loads.maneuver.legacy")
class ManeuverLoads(om.ExplicitComponent):
    """
    Computes CS25 pull-up maneuver load factors evaluated at two different load cases:

    Load case 1: with wings with almost no fuel
    Load case 2: at maximum take-off weight

    Based on formulas in :cite:`supaero:2014`, §6.3

    """

    def setup(self):
        self.add_input("data:load_case:maneuver_load_factor", val=0.0, units="unitless")
        self.add_input("data:mission:sizing:cs25:safety_factor", val=1.5, units="unitless")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")

        self.add_output("data:mission:sizing:cs25:maneuver:load_factor_1", units="unitless")
        self.add_output("data:mission:sizing:cs25:maneuver:load_factor_2", units="unitless")

    def setup_partials(self):
        self.declare_partials(
            "data:mission:sizing:cs25:maneuver:load_factor_1",
            "*",
            method="fd",
        )
        self.declare_partials(
            "data:mission:sizing:cs25:maneuver:load_factor_2",
            "*",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_maneuver = inputs["data:load_case:maneuver_load_factor"]
        sf = inputs["data:mission:sizing:cs25:safety_factor"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        if (
            n_maneuver == 0.0
        ):  # n_maneuver not provided by the user, so we evaluate it using CS 25.337
            n_maneuver = self.__n_maneuver(mtow)

        # load case #1
        n1 = sf * n_maneuver

        # load case #2
        n2 = sf * n_maneuver

        outputs["data:mission:sizing:cs25:maneuver:load_factor_1"] = n1
        outputs["data:mission:sizing:cs25:maneuver:load_factor_2"] = n2

    @staticmethod
    def __n_maneuver(w_kg):
        """
        See CS 25.337 Limit manoeuvring load factors.
        Computes the maneuver load factor based on gross weight (in kg).
        - Converts weight to pounds.
        - Uses the formula: 2 * (1 + 24000 / (w + 10000))
        - Clip the result between 2.5 and 3.8.
        """
        w = convert_units(w_kg, "kg", "lb")
        raw_n = 2 * (1 + 24000 / (w + 10000))
        return np.clip(raw_n, 2.5, 3.8)
