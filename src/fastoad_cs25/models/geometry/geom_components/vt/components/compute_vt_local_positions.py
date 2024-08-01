"""
Estimation of vertical tail chords and span
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


class ComputeVTLocalPositions(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail local positions estimation"""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:MAC:z", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_0", val=np.nan, units="rad")

        self.add_output("data:geometry:vertical_tail:tip:leading_edge:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:root:leading_edge:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:leading_edge:x:local", units="m")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:vertical_tail:tip:leading_edge:x:local", "*", method="fd"
        )
        self.declare_partials(
            "data:geometry:vertical_tail:root:leading_edge:x:local", "*", method="fd"
        )
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:leading_edge:x:local", "*", method="fd"
        )

    def compute(self, inputs, outputs):
        z_mac = inputs["data:geometry:vertical_tail:MAC:z"]
        x0_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:local"]
        mac_vt = inputs["data:geometry:vertical_tail:MAC:length"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        sweep_0_vt = inputs["data:geometry:vertical_tail:sweep_0"]

        x_mac = x0_vt - mac_vt * 0.25
        x_root = x_mac - z_mac * np.tan(sweep_0_vt)
        x_tip = x_root + b_v * np.tan(sweep_0_vt)

        outputs["data:geometry:vertical_tail:tip:leading_edge:x:local"] = x_tip
        outputs["data:geometry:vertical_tail:root:leading_edge:x:local"] = x_root
        outputs["data:geometry:vertical_tail:MAC:leading_edge:x:local"] = x_mac
