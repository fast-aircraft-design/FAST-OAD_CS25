"""
Estimation of horizontal tail mean aerodynamic chords
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


# TODO: it would be good to have a function to compute MAC for HT, VT and WING
class ComputeHTMAC(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail mean aerodynamic chord estimation"""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:center:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:length", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:y", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        root_chord = inputs["data:geometry:horizontal_tail:center:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        mac_ht = (
            (root_chord**2 + root_chord * tip_chord + tip_chord**2)
            / (tip_chord + root_chord)
            * 2
            / 3
        )

        x0_ht = (root_chord / 4) + (b_h / 6) * np.tan(sweep_25_ht) * (
            (root_chord + 2 * tip_chord) / (root_chord + tip_chord)
        )

        y0_ht = (b_h * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:horizontal_tail:MAC:length"] = mac_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"] = x0_ht
        outputs["data:geometry:horizontal_tail:MAC:y"] = y0_ht

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:center:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        n_term = root_chord**2 + root_chord * tip_chord + tip_chord**2
        d_term = root_chord + tip_chord
        m_term = 0.5 * root_chord + tip_chord
        r_term = (root_chord + 2.0 * tip_chord) / d_term
        tan_sweep = np.tan(sweep_25_ht)

        mac = "data:geometry:horizontal_tail:MAC:length"
        x0 = "data:geometry:horizontal_tail:MAC:at25percent:x:local"
        y0 = "data:geometry:horizontal_tail:MAC:y"

        partials[mac, "data:geometry:horizontal_tail:center:chord"] = (
            2.0 / 3.0 * ((2.0 * root_chord + tip_chord) * d_term - n_term) / d_term**2
        )
        partials[mac, "data:geometry:horizontal_tail:tip:chord"] = (
            2.0 / 3.0 * ((root_chord + 2.0 * tip_chord) * d_term - n_term) / d_term**2
        )

        partials[x0, "data:geometry:horizontal_tail:center:chord"] = (
            0.25 + b_h / 6.0 * tan_sweep * (-tip_chord / d_term**2)
        )
        partials[x0, "data:geometry:horizontal_tail:tip:chord"] = (
            b_h / 6.0 * tan_sweep * root_chord / d_term**2
        )
        partials[x0, "data:geometry:horizontal_tail:span"] = tan_sweep * r_term / 6.0
        partials[x0, "data:geometry:horizontal_tail:sweep_25"] = (
            b_h / 6.0 / np.cos(sweep_25_ht) ** 2 * r_term
        )

        partials[y0, "data:geometry:horizontal_tail:center:chord"] = (
            b_h / 3.0 * (0.5 * d_term - m_term) / d_term**2
        )
        partials[y0, "data:geometry:horizontal_tail:tip:chord"] = (
            b_h / 3.0 * (d_term - m_term) / d_term**2
        )
        partials[y0, "data:geometry:horizontal_tail:span"] = m_term / (3.0 * d_term)
