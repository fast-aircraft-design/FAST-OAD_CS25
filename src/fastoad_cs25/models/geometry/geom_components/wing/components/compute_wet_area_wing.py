"""
Estimation of wing wet area
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from ..constants import SERVICE_WING_GEOMETRY_WET_AREA


@oad.RegisterSubmodel(
    SERVICE_WING_GEOMETRY_WET_AREA, "fastoad.submodel.geometry.wing.wet_area.legacy"
)
class ComputeWetAreaWing(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing wet area estimation"""

    def setup(self):
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:wing:outer_area", units="m**2")
        self.add_output("data:geometry:wing:wetted_area", units="m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        wing_area = inputs["data:geometry:wing:area"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        s_pf = wing_area - 2 * l2_wing * y2_wing
        wet_area_wing = 2 * (wing_area - width_max * l2_wing)

        outputs["data:geometry:wing:outer_area"] = s_pf
        outputs["data:geometry:wing:wetted_area"] = wet_area_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        outer = "data:geometry:wing:outer_area"
        wet = "data:geometry:wing:wetted_area"

        partials[outer, "data:geometry:wing:area"] = 1.0
        partials[outer, "data:geometry:wing:root:chord"] = -2.0 * y2_wing
        partials[outer, "data:geometry:wing:root:y"] = -2.0 * l2_wing

        partials[wet, "data:geometry:wing:area"] = 2.0
        partials[wet, "data:geometry:wing:root:chord"] = -2.0 * width_max
        partials[wet, "data:geometry:fuselage:maximum_width"] = -2.0 * l2_wing
