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


class ComputeVTChords(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail chords and span estimation"""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan, units="unitless")

        self.add_output("data:geometry:vertical_tail:span", units="m")
        self.add_output("data:geometry:vertical_tail:root:chord", units="m")
        self.add_output("data:geometry:vertical_tail:tip:chord", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        lambda_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        s_v = inputs["data:geometry:vertical_tail:area"]
        taper_v = inputs["data:geometry:vertical_tail:taper_ratio"]

        b_v = np.sqrt(max(lambda_vt * s_v, 0.1))  #
        root_chord = s_v * 2 / (1 + taper_v) / b_v
        tip_chord = root_chord * taper_v

        outputs["data:geometry:vertical_tail:span"] = b_v
        outputs["data:geometry:vertical_tail:root:chord"] = root_chord
        outputs["data:geometry:vertical_tail:tip:chord"] = tip_chord

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        lambda_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        s_v = inputs["data:geometry:vertical_tail:area"]
        taper_v = inputs["data:geometry:vertical_tail:taper_ratio"]

        product = lambda_vt * s_v
        if product > 0.1:
            b_v = np.sqrt(product)
            d_b_d_area = 0.5 * lambda_vt / b_v
            d_b_d_aspect_ratio = 0.5 * s_v / b_v
        else:
            b_v = np.sqrt(0.1)
            d_b_d_area = 0.0
            d_b_d_aspect_ratio = 0.0

        denominator = (1.0 + taper_v) * b_v
        root_chord = 2.0 * s_v / denominator
        d_root_d_area = 2.0 / denominator - 2.0 * s_v / ((1.0 + taper_v) * b_v**2) * d_b_d_area
        d_root_d_taper = -2.0 * s_v / ((1.0 + taper_v) ** 2 * b_v)
        d_root_d_aspect_ratio = -2.0 * s_v / ((1.0 + taper_v) * b_v**2) * d_b_d_aspect_ratio

        span = "data:geometry:vertical_tail:span"
        root = "data:geometry:vertical_tail:root:chord"
        tip = "data:geometry:vertical_tail:tip:chord"

        partials[span, "data:geometry:vertical_tail:area"] = d_b_d_area
        partials[span, "data:geometry:vertical_tail:aspect_ratio"] = d_b_d_aspect_ratio
        partials[root, "data:geometry:vertical_tail:area"] = d_root_d_area
        partials[root, "data:geometry:vertical_tail:aspect_ratio"] = d_root_d_aspect_ratio
        partials[root, "data:geometry:vertical_tail:taper_ratio"] = d_root_d_taper
        partials[tip, "data:geometry:vertical_tail:area"] = taper_v * d_root_d_area
        partials[tip, "data:geometry:vertical_tail:aspect_ratio"] = taper_v * d_root_d_aspect_ratio
        partials[tip, "data:geometry:vertical_tail:taper_ratio"] = (
            root_chord + taper_v * d_root_d_taper
        )
