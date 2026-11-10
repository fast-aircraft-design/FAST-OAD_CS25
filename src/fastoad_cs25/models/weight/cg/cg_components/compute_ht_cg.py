"""
Estimation of horizontal tail center of gravity
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

from ..constants import SERVICE_HORIZONTAL_TAIL_CG


@oad.RegisterSubmodel(
    SERVICE_HORIZONTAL_TAIL_CG, "fastoad.submodel.weight.cg.horizontal_tail.legacy"
)
class ComputeHTcg(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail center of gravity estimation"""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:center:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input(
            "data:geometry:horizontal_tail:thickness_ratio", val=np.nan, units="unitless"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m"
        )
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:has_T_tail", val=np.nan, units="unitless")

        self.add_output("data:weight:airframe:horizontal_tail:CG:x", units="m")
        self.add_output("data:weight:airframe:horizontal_tail:CG:z", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        root_chord = inputs["data:geometry:horizontal_tail:center:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        x0_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        tail_type = np.round(inputs["data:geometry:has_T_tail"])

        tmp = root_chord * 0.25 + b_h / 2 * np.tan(sweep_25_ht / 180.0 * np.pi) - tip_chord * 0.25

        l_cg = 0.62 * (root_chord - tip_chord) + tip_chord
        x_cg_ht = 0.42 * l_cg + 0.38 * tmp
        x_cg_ht_absolute = lp_ht + fa_length - 0.25 * mac_ht + (x_cg_ht - x0_ht)

        # We assume that the upper surface of the HTP is flush with the fuselage
        z_cg_ht = height_max - root_chord * thickness_ratio / 2.0
        if tail_type == 1:
            z_cg_ht += inputs["data:geometry:vertical_tail:span"]

        outputs["data:weight:airframe:horizontal_tail:CG:x"] = x_cg_ht_absolute
        outputs["data:weight:airframe:horizontal_tail:CG:z"] = z_cg_ht

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:center:chord"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        tail_type = np.round(inputs["data:geometry:has_T_tail"])

        x_output = "data:weight:airframe:horizontal_tail:CG:x"
        z_output = "data:weight:airframe:horizontal_tail:CG:z"

        sweep_rad = sweep_25_ht / 180.0 * np.pi
        partials[x_output, "data:geometry:horizontal_tail:center:chord"] = 0.3554
        partials[x_output, "data:geometry:horizontal_tail:tip:chord"] = 0.0646
        partials[x_output, "data:geometry:horizontal_tail:span"] = 0.19 * np.tan(sweep_rad)
        partials[x_output, "data:geometry:horizontal_tail:sweep_25"] = (
            0.19 * b_h * np.pi / 180.0 / np.cos(sweep_rad) ** 2
        )
        partials[x_output, "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = 1.0
        partials[x_output, "data:geometry:wing:MAC:at25percent:x"] = 1.0
        partials[x_output, "data:geometry:horizontal_tail:MAC:length"] = -0.25
        partials[x_output, "data:geometry:horizontal_tail:MAC:at25percent:x:local"] = -1.0

        partials[z_output, "data:geometry:fuselage:maximum_height"] = 1.0
        partials[z_output, "data:geometry:horizontal_tail:center:chord"] = -thickness_ratio / 2.0
        partials[z_output, "data:geometry:horizontal_tail:thickness_ratio"] = -root_chord / 2.0
        partials[z_output, "data:geometry:vertical_tail:span"] = 1.0 if tail_type == 1 else 0.0
