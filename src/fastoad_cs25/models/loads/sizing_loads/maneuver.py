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
import openmdao.api as om

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
        self.add_input("data:load_case:maneuver_load_factor", val=2.5)
        self.add_input("data:mission:sizing:cs25:safety_factor", val=1.5)

        self.add_output("data:mission:sizing:cs25:maneuver:load_factor_1")
        self.add_output("data:mission:sizing:cs25:maneuver:load_factor_2")

    def setup_partials(self):
        self.declare_partials(
            "data:mission:sizing:cs25:maneuver:load_factor_1",
            ["data:load_case:maneuver_load_factor", "data:mission:sizing:cs25:safety_factor"],
            method="fd",
        )
        self.declare_partials(
            "data:mission:sizing:cs25:maneuver:load_factor_2",
            ["data:load_case:maneuver_load_factor", "data:mission:sizing:cs25:safety_factor"],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_maneuver = inputs["data:load_case:maneuver_load_factor"]
        sf = inputs["data:mission:sizing:cs25:safety_factor"]

        # load case #1
        n1 = sf * n_maneuver

        # load case #2
        n2 = sf * n_maneuver

        outputs["data:mission:sizing:cs25:maneuver:load_factor_1"] = n1
        outputs["data:mission:sizing:cs25:maneuver:load_factor_2"] = n2
