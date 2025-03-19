"""
Estimation of center wing chord (l1')
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

import numpy as np
import openmdao.api as om
from scipy.interpolate import make_interp_spline


class ComputeCenterChord(om.ExplicitComponent):
    """
    Estimation of center wing chord (l1') by extrapolation of the virtual edges of the wing.
    See Figure 3-3bis of :cite:`supaero:2014`.
    """

    def setup(self):
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")

        self.add_output("data:geometry:wing:center:chord", units="m")
        self.add_output("data:geometry:wing:center:leading_edge:x:local", units="m")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:wing:center:chord",
            [
                "data:geometry:wing:root:chord",
                "data:geometry:wing:root:y",
                "data:geometry:wing:kink:chord",
                "data:geometry:wing:kink:y",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:tip:y",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:wing:center:leading_edge:x:local",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:kink:leading_edge:x:local",
                "data:geometry:wing:kink:y",
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:tip:y",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        l3_wing = inputs["data:geometry:wing:kink:chord"]
        x3_wing = inputs["data:geometry:wing:kink:leading_edge:x:local"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        if y3_wing <= y2_wing:
            y_values = np.squeeze([y2_wing, y4_wing])
            x_values = np.squeeze([[0.0], x4_wing])
            l_values = np.squeeze([l2_wing, l4_wing])
        else:
            y_values = np.squeeze([y2_wing, y3_wing, y4_wing])
            x_values = np.squeeze([[0.0], x3_wing, x4_wing])
            l_values = np.squeeze([l2_wing, l3_wing, l4_wing])

        # Linear interpolation to evaluate the central chord
        line_x_leading_edge = make_interp_spline(y_values, x_values, k=1)
        line_l_center = make_interp_spline(y_values, l_values, k=1)

        outputs["data:geometry:wing:center:chord"] = line_l_center(0.0)
        outputs["data:geometry:wing:center:leading_edge:x:local"] = line_x_leading_edge(0.0)
