"""
Estimation of horizontal tail chords and span
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


class ComputeHTLocalPositions(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail local positions estimation"""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:MAC:y", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_0", val=np.nan, units="rad")

        self.add_output("data:geometry:horizontal_tail:tip:leading_edge:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:center:leading_edge:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:leading_edge:x:local", units="m")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:horizontal_tail:tip:leading_edge:x:local", "*", method="fd"
        )
        self.declare_partials(
            "data:geometry:horizontal_tail:center:leading_edge:x:local", "*", method="fd"
        )
        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:leading_edge:x:local", "*", method="fd"
        )

    def compute(self, inputs, outputs):
        y_mac = inputs["data:geometry:horizontal_tail:MAC:y"]
        x0_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        sweep_0_ht = inputs["data:geometry:horizontal_tail:sweep_0"]

        x_mac = x0_ht - mac_ht * 0.25
        x_root = x_mac - y_mac * np.tan(sweep_0_ht)
        x_tip = x_root + b_h / 2 * np.tan(sweep_0_ht)

        outputs["data:geometry:horizontal_tail:tip:leading_edge:x:local"] = x_tip
        outputs["data:geometry:horizontal_tail:center:leading_edge:x:local"] = x_root
        outputs["data:geometry:horizontal_tail:MAC:leading_edge:x:local"] = x_mac
