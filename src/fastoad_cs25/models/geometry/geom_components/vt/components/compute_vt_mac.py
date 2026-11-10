"""
Estimation of vertical tail mean aerodynamic chords
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
class ComputeVTMAC(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail mean aerodynamic chord estimation"""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        mac_vt = (
            (root_chord**2 + root_chord * tip_chord + tip_chord**2)
            / (tip_chord + root_chord)
            * 2.0
            / 3.0
        )

        x0_vt = (root_chord / 4) + (b_v / 3) * np.tan(sweep_25_vt) * (
            (root_chord + 2 * tip_chord) / (root_chord + tip_chord)
        )

        z0_vt = (2 * b_v * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt
        outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        n_term = root_chord**2 + root_chord * tip_chord + tip_chord**2
        d_term = root_chord + tip_chord
        m_term = 0.5 * root_chord + tip_chord
        r_term = (root_chord + 2.0 * tip_chord) / d_term
        tan_sweep = np.tan(sweep_25_vt)

        mac = "data:geometry:vertical_tail:MAC:length"
        x0 = "data:geometry:vertical_tail:MAC:at25percent:x:local"
        z0 = "data:geometry:vertical_tail:MAC:z"

        partials[mac, "data:geometry:vertical_tail:root:chord"] = (
            2.0 / 3.0 * ((2.0 * root_chord + tip_chord) * d_term - n_term) / d_term**2
        )
        partials[mac, "data:geometry:vertical_tail:tip:chord"] = (
            2.0 / 3.0 * ((root_chord + 2.0 * tip_chord) * d_term - n_term) / d_term**2
        )

        partials[x0, "data:geometry:vertical_tail:root:chord"] = 0.25 + b_v / 3.0 * tan_sweep * (
            -tip_chord / d_term**2
        )
        partials[x0, "data:geometry:vertical_tail:tip:chord"] = (
            b_v / 3.0 * tan_sweep * root_chord / d_term**2
        )
        partials[x0, "data:geometry:vertical_tail:span"] = tan_sweep * r_term / 3.0
        partials[x0, "data:geometry:vertical_tail:sweep_25"] = (
            b_v / 3.0 / np.cos(sweep_25_vt) ** 2 * r_term
        )

        partials[z0, "data:geometry:vertical_tail:root:chord"] = (
            2.0 * b_v / 3.0 * (0.5 * d_term - m_term) / d_term**2
        )
        partials[z0, "data:geometry:vertical_tail:tip:chord"] = (
            2.0 * b_v / 3.0 * (d_term - m_term) / d_term**2
        )
        partials[z0, "data:geometry:vertical_tail:span"] = 2.0 * m_term / (3.0 * d_term)
