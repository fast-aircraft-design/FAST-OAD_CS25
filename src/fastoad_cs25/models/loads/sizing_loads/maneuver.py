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
from scipy.constants import g

from ..constants import SERVICE_MANEUVER_LOADS


@oad.RegisterSubmodel(SERVICE_MANEUVER_LOADS, "fastoad.submodel.loads.maneuver.legacy")
class ManeuverLoads(om.ExplicitComponent):
    """
    Computes CS25 pull-up maneuver evaluated at two different load cases:

    Load case 1: with wings with almost no fuel
    Load case 2: at maximum take-off weight

    Based on formulas in :cite:`supaero:2014`, §6.3

    """

    def initialize(self):
        self.options.declare(
            "fuel_load_alleviation",
            types=bool,
            default=True,
            desc="If False this simulates a dry wing,"
            "i.e. the sizing load 2 does not take into account the fuel weight.",
        )

    def setup(self):
        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:load_case:maneuver_load_factor", val=2.5)
        self.add_input("data:mission:sizing:cs25:safety_factor_1", val=1.5)
        self.add_input("data:mission:sizing:cs25:safety_factor_2", val=1.5)

        self.add_output("data:mission:sizing:cs25:maneuver:load_factor_1")
        self.add_output("data:mission:sizing:cs25:maneuver:load_factor_2")
        self.add_output("data:mission:sizing:cs25:maneuver:sizing_load_1", units="N")
        self.add_output("data:mission:sizing:cs25:maneuver:sizing_load_2", units="N")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mzfw = inputs["data:weight:aircraft:MZFW"]
        mfw = inputs["data:weight:aircraft:MFW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        n_maneuver = inputs["data:load_case:maneuver_load_factor"]
        sf1 = inputs["data:mission:sizing:cs25:safety_factor_1"]
        sf2 = inputs["data:mission:sizing:cs25:safety_factor_2"]

        # load case #1
        m1 = 1.05 * mzfw
        n1 = sf1 * n_maneuver
        n1m1 = n1 * m1 * g

        # load case #2
        n2 = sf2 * n_maneuver

        if not self.options["fuel_load_alleviation"]:
            n2m2 = n2 * mtow * g
        else:
            mcv = min(0.8 * mfw, mtow - mzfw)  # fuel mass in the overhanging part of wing
            n2m2 = n2 * (mtow - 0.55 * mcv) * g

        outputs["data:mission:sizing:cs25:maneuver:load_factor_1"] = n1
        outputs["data:mission:sizing:cs25:maneuver:load_factor_2"] = n2
        outputs["data:mission:sizing:cs25:maneuver:sizing_load_1"] = n1m1
        outputs["data:mission:sizing:cs25:maneuver:sizing_load_2"] = n2m2
