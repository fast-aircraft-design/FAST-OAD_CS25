"""
Estimation of wing Ys (sections span)
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


class ComputeYWing(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing Ys estimation"""

    def initialize(self):
        self.options.declare("impose_absolute_kink", types=bool, default=False)

    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        if self.options["impose_absolute_kink"]:
            self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        else:
            self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan, units="unitless")

        self.add_output("data:geometry:wing:span", units="m")
        self.add_output("data:geometry:wing:root:y", units="m")
        if not self.options["impose_absolute_kink"]:
            self.add_output("data:geometry:wing:kink:y", units="m")
        self.add_output("data:geometry:wing:tip:y", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        span = np.sqrt(lambda_wing * wing_area)

        # Wing geometry
        y4_wing = span / 2.0
        y2_wing = width_max / 2.0

        outputs["data:geometry:wing:span"] = span
        outputs["data:geometry:wing:root:y"] = y2_wing
        outputs["data:geometry:wing:tip:y"] = y4_wing

        if not self.options["impose_absolute_kink"]:
            wing_break = inputs["data:geometry:wing:kink:span_ratio"]
            y3_wing = np.maximum(y2_wing, y4_wing * wing_break)
            outputs["data:geometry:wing:kink:y"] = y3_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        span = np.sqrt(lambda_wing * wing_area)
        d_span_d_aspect_ratio = 0.5 * wing_area / span
        d_span_d_area = 0.5 * lambda_wing / span

        partials["data:geometry:wing:span", "data:geometry:wing:aspect_ratio"] = (
            d_span_d_aspect_ratio
        )
        partials["data:geometry:wing:span", "data:geometry:wing:area"] = d_span_d_area
        partials["data:geometry:wing:root:y", "data:geometry:fuselage:maximum_width"] = 0.5
        partials["data:geometry:wing:tip:y", "data:geometry:wing:aspect_ratio"] = (
            0.5 * d_span_d_aspect_ratio
        )
        partials["data:geometry:wing:tip:y", "data:geometry:wing:area"] = 0.5 * d_span_d_area

        if not self.options["impose_absolute_kink"]:
            wing_break = inputs["data:geometry:wing:kink:span_ratio"]
            y4_wing = span / 2.0
            y2_wing = width_max / 2.0
            if y4_wing * wing_break >= y2_wing:
                partials["data:geometry:wing:kink:y", "data:geometry:wing:aspect_ratio"] = (
                    wing_break * 0.5 * d_span_d_aspect_ratio
                )
                partials["data:geometry:wing:kink:y", "data:geometry:wing:area"] = (
                    wing_break * 0.5 * d_span_d_area
                )
                partials["data:geometry:wing:kink:y", "data:geometry:wing:kink:span_ratio"] = (
                    y4_wing
                )
            else:
                partials["data:geometry:wing:kink:y", "data:geometry:fuselage:maximum_width"] = 0.5
