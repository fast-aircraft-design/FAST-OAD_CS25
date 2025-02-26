"""
Estimation of relative kink from an absolute value
"""

#  This file is part of FAST-OAD_CS25
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


class ComputeRelativeKink(om.ExplicitComponent):
    """Estimation of relative kink from an absolute value"""

    def setup(self):
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")

        self.add_output("data:geometry:wing:kink:span_ratio")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:wing:kink:span_ratio",
            [
                "data:geometry:wing:kink:y",
                "data:geometry:wing:tip:y",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs):
        y3_wing = inputs["data:geometry:wing:kink:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        if y4_wing > 0:
            wing_break = y3_wing / y4_wing
        else:
            wing_break = 0.0

        outputs["data:geometry:wing:kink:span_ratio"] = wing_break
