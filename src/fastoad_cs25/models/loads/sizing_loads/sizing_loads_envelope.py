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

import openmdao.api as om
from fastoad.module_management.service_registry import RegisterSubmodel

from ..constants import SERVICE_SIZING_LOADS_ENVELOPE


@RegisterSubmodel(SERVICE_SIZING_LOADS_ENVELOPE, "fastoad.submodel.loads.envelope.legacy")
class SizingLoadsEnvelope(om.ExplicitComponent):
    """
    Computes CS25 sizing loading evaluated at two different load cases:

    Load case 1: with wings with almost no fuel
    Load case 2: at maximum take-off weight

    Based on formulas in :cite:`supaero:2014`, §6.3

    """

    def setup(self):
        self.add_input("data:mission:sizing:cs25:gust:load_factor_1")
        self.add_input("data:mission:sizing:cs25:gust:load_factor_2")
        self.add_input("data:mission:sizing:cs25:gust:sizing_load_1", units="N")
        self.add_input("data:mission:sizing:cs25:gust:sizing_load_2", units="N")
        self.add_input("data:mission:sizing:cs25:maneuver:load_factor_1")
        self.add_input("data:mission:sizing:cs25:maneuver:load_factor_2")
        self.add_input("data:mission:sizing:cs25:maneuver:sizing_load_1", units="N")
        self.add_input("data:mission:sizing:cs25:maneuver:sizing_load_2", units="N")

        self.add_output("data:mission:sizing:cs25:load_factor_1")
        self.add_output("data:mission:sizing:cs25:load_factor_2")
        self.add_output("data:mission:sizing:cs25:sizing_load_1", units="N")
        self.add_output("data:mission:sizing:cs25:sizing_load_2", units="N")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n1_gust = inputs["data:mission:sizing:cs25:gust:load_factor_1"]
        n2_gust = inputs["data:mission:sizing:cs25:gust:load_factor_2"]
        n1m1_gust = inputs["data:mission:sizing:cs25:gust:sizing_load_1"]
        n2m2_gust = inputs["data:mission:sizing:cs25:gust:sizing_load_2"]
        n1_maneuver = inputs["data:mission:sizing:cs25:maneuver:load_factor_1"]
        n2_maneuver = inputs["data:mission:sizing:cs25:maneuver:load_factor_2"]
        n1m1_maneuver = inputs["data:mission:sizing:cs25:maneuver:sizing_load_1"]
        n2m2_maneuver = inputs["data:mission:sizing:cs25:maneuver:sizing_load_2"]

        n1 = max(n1_gust, n1_maneuver)
        n2 = max(n2_gust, n2_maneuver)
        n1m1 = max(n1m1_gust, n1m1_maneuver)
        n2m2 = max(n2m2_gust, n2m2_maneuver)

        outputs["data:mission:sizing:cs25:load_factor_1"] = n1
        outputs["data:mission:sizing:cs25:load_factor_2"] = n2
        outputs["data:mission:sizing:cs25:sizing_load_1"] = n1m1
        outputs["data:mission:sizing:cs25:sizing_load_2"] = n2m2
