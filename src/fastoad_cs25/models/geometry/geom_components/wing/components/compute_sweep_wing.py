"""
    Estimation of wing sweeps
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2024 ONERA & ISAE-SUPAERO
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

import math

import numpy as np
import openmdao.api as om


class ComputeSweepWing(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing sweeps estimation"""

    def setup(self):
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_100_ratio", val=0.0, units=None)

        self.add_output("data:geometry:wing:sweep_0", units="deg")
        self.add_output("data:geometry:wing:sweep_100_inner", val=0.0, units="deg")
        self.add_output("data:geometry:wing:sweep_100_outer", units="deg")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:wing:sweep_0",
            [
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:root:y",
                "data:geometry:wing:kink:y",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:wing:sweep_100_inner",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:root:virtual_chord",
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:tip:chord",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:wing:sweep_100_outer",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:root:virtual_chord",
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:tip:chord",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs):
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        outputs["data:geometry:wing:sweep_100_outer"] = (
            math.atan((x4_wing + l4_wing - l1_wing) / (y4_wing - y2_wing)) / math.pi * 180.0
        )
        outputs["data:geometry:wing:sweep_0"] = (
            math.atan(x4_wing / (y4_wing - y2_wing)) / math.pi * 180.0
        )
        if y3_wing == y2_wing:
            ratio = 1.0
        else:
            ratio = inputs["data:geometry:wing:sweep_100_ratio"]

        outputs["data:geometry:wing:sweep_100_inner"] = (
            outputs["data:geometry:wing:sweep_100_outer"] * ratio
        )
