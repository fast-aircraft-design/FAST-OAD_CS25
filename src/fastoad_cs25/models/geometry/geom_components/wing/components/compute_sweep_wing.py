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

        self.add_output("data:geometry:wing:sweep_0", units="deg")
        self.add_output("data:geometry:wing:sweep_100_outer", units="deg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        outputs["data:geometry:wing:sweep_100_outer"] = (
            np.atan((x4_wing + l4_wing - l1_wing) / (y4_wing - y2_wing)) / np.pi * 180.0
        )
        outputs["data:geometry:wing:sweep_0"] = (
            np.atan(x4_wing / (y4_wing - y2_wing)) / np.pi * 180.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        v = y4_wing - y2_wing
        u0 = x4_wing
        u1 = x4_wing + l4_wing - l1_wing
        factor = 180.0 / np.pi

        d0_du = factor * v / (u0**2 + v**2)
        d0_dv = -factor * u0 / (u0**2 + v**2)
        d1_du = factor * v / (u1**2 + v**2)
        d1_dv = -factor * u1 / (u1**2 + v**2)

        sweep_0 = "data:geometry:wing:sweep_0"
        sweep_100 = "data:geometry:wing:sweep_100_outer"

        partials[sweep_0, "data:geometry:wing:tip:leading_edge:x:local"] = d0_du
        partials[sweep_0, "data:geometry:wing:tip:y"] = d0_dv
        partials[sweep_0, "data:geometry:wing:root:y"] = -d0_dv

        partials[sweep_100, "data:geometry:wing:tip:leading_edge:x:local"] = d1_du
        partials[sweep_100, "data:geometry:wing:tip:chord"] = d1_du
        partials[sweep_100, "data:geometry:wing:root:virtual_chord"] = -d1_du
        partials[sweep_100, "data:geometry:wing:tip:y"] = d1_dv
        partials[sweep_100, "data:geometry:wing:root:y"] = -d1_dv


class ComputeInnerSweepWing(om.ExplicitComponent):
    """Inner Wing sweep estimation"""

    def setup(self):
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_100_outer", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:sweep_100_ratio", val=0.0, units="unitless")

        self.add_output("data:geometry:wing:sweep_100_inner", val=0.0, units="deg")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:wing:sweep_100_inner",
            ["data:geometry:wing:sweep_100_outer", "data:geometry:wing:sweep_100_ratio"],
            method="exact",
        )

    def compute(self, inputs, outputs):
        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        sweep_100_outer = inputs["data:geometry:wing:sweep_100_outer"]
        if y3_wing == y2_wing:
            ratio = 1.0
        else:
            ratio = inputs["data:geometry:wing:sweep_100_ratio"]

        outputs["data:geometry:wing:sweep_100_inner"] = sweep_100_outer * ratio

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        sweep_100_outer = inputs["data:geometry:wing:sweep_100_outer"]
        sweep_100_ratio = inputs["data:geometry:wing:sweep_100_ratio"]

        output_name = "data:geometry:wing:sweep_100_inner"
        if y3_wing == y2_wing:
            partials[output_name, "data:geometry:wing:sweep_100_outer"] = 1.0
            partials[output_name, "data:geometry:wing:sweep_100_ratio"] = 0.0
        else:
            partials[output_name, "data:geometry:wing:sweep_100_outer"] = sweep_100_ratio
            partials[output_name, "data:geometry:wing:sweep_100_ratio"] = sweep_100_outer
