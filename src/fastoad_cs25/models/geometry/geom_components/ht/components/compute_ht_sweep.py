"""
Estimation of horizontal tail sweeps
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


class ComputeHTSweep(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail sweeps estimation"""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:center:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")

        self.add_output("data:geometry:horizontal_tail:sweep_0", units="deg")
        self.add_output("data:geometry:horizontal_tail:sweep_100", units="deg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:center:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0

        # The conversion to degrees is intentionally done here and not in OpenMDAO, so that
        # the outputs will naturally be displayed in degrees.
        sweep_0_ht = np.degrees(
            (np.atan(np.tan(sweep_25_ht) + 0.25 * (root_chord - tip_chord) / half_span))
        )
        sweep_100_ht = np.degrees(
            (np.atan(np.tan(sweep_25_ht) - 0.75 * (root_chord - tip_chord) / half_span))
        )

        outputs["data:geometry:horizontal_tail:sweep_0"] = sweep_0_ht
        outputs["data:geometry:horizontal_tail:sweep_100"] = sweep_100_ht

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:center:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0
        delta_chord = root_chord - tip_chord
        tan_sweep = np.tan(sweep_25_ht)
        factor = 180.0 / np.pi
        d_tan_d_sweep = 1.0 / np.cos(sweep_25_ht) ** 2

        sweep_0 = "data:geometry:horizontal_tail:sweep_0"
        sweep_100 = "data:geometry:horizontal_tail:sweep_100"

        for output_name, coefficient in ((sweep_0, 0.25), (sweep_100, -0.75)):
            argument = tan_sweep + coefficient * delta_chord / half_span
            d_output_d_argument = factor / (1.0 + argument**2)
            d_argument_d_root = coefficient / half_span
            d_argument_d_tip = -coefficient / half_span
            d_argument_d_span = -coefficient * delta_chord / (2.0 * half_span**2)

            partials[output_name, "data:geometry:horizontal_tail:center:chord"] = (
                d_output_d_argument * d_argument_d_root
            )
            partials[output_name, "data:geometry:horizontal_tail:tip:chord"] = (
                d_output_d_argument * d_argument_d_tip
            )
            partials[output_name, "data:geometry:horizontal_tail:span"] = (
                d_output_d_argument * d_argument_d_span
            )
            partials[output_name, "data:geometry:horizontal_tail:sweep_25"] = (
                d_output_d_argument * d_tan_d_sweep
            )
