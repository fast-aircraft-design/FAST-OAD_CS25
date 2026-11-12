"""
Estimation of vertical tail sweeps
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


# TODO: HT and VT components are similar --> factorize
class ComputeVTSweep(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail sweeps estimation"""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")

        self.add_output("data:geometry:vertical_tail:sweep_0", units="deg")
        self.add_output("data:geometry:vertical_tail:sweep_100", units="deg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        sweep_0_vt = (
            (
                np.pi / 2
                - np.atan(
                    b_v
                    / (
                        0.25 * root_chord
                        - 0.25 * tip_chord
                        + b_v * np.tan(sweep_25_vt / 180.0 * np.pi)
                    )
                )
            )
            / np.pi
            * 180.0
        )
        sweep_100_vt = (
            (
                np.pi / 2
                - np.atan(
                    b_v
                    / (
                        b_v * np.tan(sweep_25_vt / 180.0 * np.pi)
                        - 0.75 * root_chord
                        + 0.75 * tip_chord
                    )
                )
            )
            / np.pi
            * 180.0
        )

        outputs["data:geometry:vertical_tail:sweep_0"] = sweep_0_vt
        outputs["data:geometry:vertical_tail:sweep_100"] = sweep_100_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_v = inputs["data:geometry:vertical_tail:span"]
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]

        sweep_rad = sweep_25_vt / 180.0 * np.pi
        tan_sweep = np.tan(sweep_rad)
        delta_chord = root_chord - tip_chord
        factor = 180.0 / np.pi
        d_argument_d_sweep = b_v * np.pi / 180.0 / np.cos(sweep_rad) ** 2

        configurations = (
            (
                "data:geometry:vertical_tail:sweep_0",
                0.25 * delta_chord + b_v * tan_sweep,
                0.25,
                -0.25,
            ),
            (
                "data:geometry:vertical_tail:sweep_100",
                b_v * tan_sweep - 0.75 * delta_chord,
                -0.75,
                0.75,
            ),
        )

        for output_name, argument, d_argument_d_root, d_argument_d_tip in configurations:
            denominator = b_v**2 + argument**2
            d_theta_d_argument = b_v / denominator
            partials[output_name, "data:geometry:vertical_tail:span"] = factor * (
                -argument / denominator + d_theta_d_argument * tan_sweep
            )
            partials[output_name, "data:geometry:vertical_tail:root:chord"] = (
                factor * d_theta_d_argument * d_argument_d_root
            )
            partials[output_name, "data:geometry:vertical_tail:tip:chord"] = (
                factor * d_theta_d_argument * d_argument_d_tip
            )
            partials[output_name, "data:geometry:vertical_tail:sweep_25"] = (
                factor * d_theta_d_argument * d_argument_d_sweep
            )
