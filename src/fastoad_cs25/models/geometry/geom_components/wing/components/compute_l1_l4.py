"""
Estimation of wing chords (l1 and l4)
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


class ComputeL1AndL4Wing(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing chords (l1 and l4) estimation"""

    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:virtual_taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_100_inner", val=0.0, units="rad")

        self.add_output("data:geometry:wing:root:virtual_chord", units="m")
        self.add_output("data:geometry:wing:tip:chord", units="m")

    def setup_partials(self):
        self.declare_partials("data:geometry:wing:root:virtual_chord", "*", method="fd")
        self.declare_partials("data:geometry:wing:tip:chord", "*", method="fd")

    def compute(self, inputs, outputs):
        wing_area = inputs["data:geometry:wing:area"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        span = 2 * y4_wing
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        sweep_100 = inputs["data:geometry:wing:sweep_100_inner"]
        virtual_taper_ratio = inputs["data:geometry:wing:virtual_taper_ratio"]

        l1_wing = (
            wing_area
            - (y3_wing - y2_wing) * (y3_wing + y2_wing) * (math.tan(sweep_25) - math.tan(sweep_100))
        ) / (
            (1.0 + virtual_taper_ratio) / 2.0 * (span - 2 * y2_wing)
            + 2 * y2_wing
            - (3.0 * (1.0 - virtual_taper_ratio) * (y3_wing - y2_wing) * (y3_wing + y2_wing))
            / (2.0 * (span - 2 * y2_wing))
        )

        l4_wing = l1_wing * virtual_taper_ratio

        outputs["data:geometry:wing:root:virtual_chord"] = l1_wing
        outputs["data:geometry:wing:tip:chord"] = l4_wing
