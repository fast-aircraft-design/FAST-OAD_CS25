"""
Estimation of vertical tail distance
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

import numpy as np
import openmdao.api as om


class ComputeVTDistance(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail distance estimation"""

    def setup(self):
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "settings:geometry:vertical_tail:position_ratio_on_fuselage", val=0.88, units="unitless"
        )

        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            [
                "data:geometry:fuselage:length",
                "data:geometry:wing:MAC:at25percent:x",
                "settings:geometry:vertical_tail:position_ratio_on_fuselage",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fus_length = inputs["data:geometry:fuselage:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        vtp_aero_center_ratio = inputs["settings:geometry:vertical_tail:position_ratio_on_fuselage"]

        lp_vt = vtp_aero_center_ratio * fus_length - fa_length

        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = lp_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fus_length = inputs["data:geometry:fuselage:length"]
        vtp_aero_center_ratio = inputs["settings:geometry:vertical_tail:position_ratio_on_fuselage"]

        output_name = "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"
        partials[output_name, "data:geometry:fuselage:length"] = vtp_aero_center_ratio
        partials[output_name, "settings:geometry:vertical_tail:position_ratio_on_fuselage"] = (
            fus_length
        )
        partials[output_name, "data:geometry:wing:MAC:at25percent:x"] = -1.0
