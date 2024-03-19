"""
    Estimation of wing ToC
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from ..constants import SERVICE_WING_GEOMETRY_THICKNESS


@oad.RegisterSubmodel(
    SERVICE_WING_GEOMETRY_THICKNESS, "fastoad.submodel.geometry.wing.thickness.legacy"
)
class ComputeToCWing(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing ToC estimation"""

    def setup(self):
        self.add_input("data:TLAR:cruise_mach", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan)

        self.add_output("data:geometry:wing:thickness_ratio")
        self.add_output("data:geometry:wing:root:thickness_ratio")
        self.add_output("data:geometry:wing:kink:thickness_ratio")
        self.add_output("data:geometry:wing:tip:thickness_ratio")

    def setup_partials(self):
        self.declare_partials("data:geometry:wing:thickness_ratio", "*", method="fd")
        self.declare_partials("data:geometry:wing:root:thickness_ratio", "*", method="fd")
        self.declare_partials("data:geometry:wing:kink:thickness_ratio", "*", method="fd")
        self.declare_partials("data:geometry:wing:tip:thickness_ratio", "*", method="fd")

    def compute(self, inputs, outputs):
        cruise_mach = inputs["data:TLAR:cruise_mach"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        wing_break = inputs["data:geometry:wing:kink:span_ratio"]

        # Relative thickness
        el_aero = 0.89 - (cruise_mach + 0.02) * math.sqrt(math.cos(sweep_25 / 180.0 * math.pi))
        el_emp = 1.24 * el_aero
        if wing_break == 0.0:
            el_break = el_emp  # Kink is set on root chord
        else:
            el_break = 0.94 * el_aero
        el_ext = 0.86 * el_aero

        outputs["data:geometry:wing:thickness_ratio"] = el_aero
        outputs["data:geometry:wing:root:thickness_ratio"] = el_emp
        outputs["data:geometry:wing:kink:thickness_ratio"] = el_break
        outputs["data:geometry:wing:tip:thickness_ratio"] = el_ext
