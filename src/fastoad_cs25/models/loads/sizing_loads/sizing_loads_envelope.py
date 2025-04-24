"""
Python module for the computation of maneuver and gust sizing load case envelope.
"""
#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
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
import openmdao.api as om
from scipy.constants import g

from ..constants import SERVICE_SIZING_LOADS_ENVELOPE


@oad.RegisterSubmodel(SERVICE_SIZING_LOADS_ENVELOPE, "fastoad.submodel.loads.envelope.legacy")
class SizingLoadsEnvelope(om.ExplicitComponent):
    """
    Computes CS25 sizing loading evaluated at two different load cases:

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
        self.add_input("data:mission:sizing:cs25:gust:load_factor_1")
        self.add_input("data:mission:sizing:cs25:gust:load_factor_2")
        self.add_input("data:mission:sizing:cs25:maneuver:load_factor_1")
        self.add_input("data:mission:sizing:cs25:maneuver:load_factor_2")

        self.add_output("data:mission:sizing:cs25:envelope:max_load_factor_1")
        self.add_output("data:mission:sizing:cs25:envelope:max_load_factor_2")
        self.add_output("data:mission:sizing:cs25:envelope:max_sizing_load_1", units="N")
        self.add_output("data:mission:sizing:cs25:envelope:max_sizing_load_2", units="N")

    def setup_partials(self):
        self.declare_partials(
            "data:mission:sizing:cs25:envelope:max_load_factor_1",
            [
                "data:mission:sizing:cs25:gust:load_factor_1",
                "data:mission:sizing:cs25:maneuver:load_factor_1",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:mission:sizing:cs25:envelope:max_load_factor_2",
            [
                "data:mission:sizing:cs25:gust:load_factor_2",
                "data:mission:sizing:cs25:maneuver:load_factor_2",
            ],
            method="fd",
        )

        self.declare_partials(
            "data:mission:sizing:cs25:envelope:max_sizing_load_1",
            [
                "data:weight:aircraft:MZFW",
                "data:weight:aircraft:MFW",
                "data:weight:aircraft:MTOW",
                "data:mission:sizing:cs25:gust:load_factor_1",
                "data:mission:sizing:cs25:maneuver:load_factor_1",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:mission:sizing:cs25:envelope:max_sizing_load_2",
            [
                "data:weight:aircraft:MZFW",
                "data:weight:aircraft:MFW",
                "data:weight:aircraft:MTOW",
                "data:mission:sizing:cs25:gust:load_factor_2",
                "data:mission:sizing:cs25:maneuver:load_factor_2",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mzfw = inputs["data:weight:aircraft:MZFW"]
        mfw = inputs["data:weight:aircraft:MFW"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        n1_maneuver = inputs["data:mission:sizing:cs25:maneuver:load_factor_1"]
        n2_maneuver = inputs["data:mission:sizing:cs25:maneuver:load_factor_2"]
        n1_gust = inputs["data:mission:sizing:cs25:gust:load_factor_1"]
        n2_gust = inputs["data:mission:sizing:cs25:gust:load_factor_2"]

        # load case #1
        m1 = 1.05 * mzfw
        n1m1_maneuver = n1_maneuver * m1 * g
        n1m1_gust = n1_gust * m1 * g

        # load case #2
        if not self.options["fuel_load_alleviation"]:
            n2m2_maneuver = n2_maneuver * mtow * g
            n2m2_gust = n2_gust * mtow * g
        else:
            mcv = min(0.8 * mfw, mtow - mzfw)  # fuel mass in the overhanging part of wing
            n2m2_maneuver = n2_maneuver * (mtow - 0.55 * mcv) * g
            n2m2_gust = n2_gust * (mtow - 0.55 * mcv) * g

        n1 = max(n1_gust, n1_maneuver)
        n2 = max(n2_gust, n2_maneuver)
        n1m1 = max(n1m1_gust, n1m1_maneuver)
        n2m2 = max(n2m2_gust, n2m2_maneuver)

        outputs["data:mission:sizing:cs25:envelope:max_load_factor_1"] = n1
        outputs["data:mission:sizing:cs25:envelope:max_load_factor_2"] = n2
        outputs["data:mission:sizing:cs25:envelope:max_sizing_load_1"] = n1m1
        outputs["data:mission:sizing:cs25:envelope:max_sizing_load_2"] = n2m2
