"""
Estimation of max fuel weight
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

from ..constants import SERVICE_WING_GEOMETRY_MFW


# TODO: This belongs more to mass breakdown than geometry
@oad.RegisterSubmodel(SERVICE_WING_GEOMETRY_MFW, "fastoad.submodel.geometry.wing.mfw.legacy")
class ComputeMFW(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Max fuel weight estimation"""

    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan, units="unitless")

        self.add_output("data:weight:aircraft:MFW", units="kg")

    def setup_partials(self):
        self.declare_partials("data:weight:aircraft:MFW", "*", method="exact")

    def compute(self, inputs, outputs):
        wing_area = inputs["data:geometry:wing:area"]
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        el_emp = inputs["data:geometry:wing:root:thickness_ratio"]
        el_ext = inputs["data:geometry:wing:tip:thickness_ratio"]

        # TODO: remove hard coded value
        mfw = 224 * (wing_area**1.5 * lambda_wing ** (-0.4) * (0.6 * el_emp + 0.4 * el_ext)) + 1570

        outputs["data:weight:aircraft:MFW"] = mfw

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        el_emp = inputs["data:geometry:wing:root:thickness_ratio"]
        el_ext = inputs["data:geometry:wing:tip:thickness_ratio"]

        thickness_term = 0.6 * el_emp + 0.4 * el_ext
        base = wing_area**1.5 * lambda_wing ** (-0.4)

        output_name = "data:weight:aircraft:MFW"
        partials[output_name, "data:geometry:wing:area"] = (
            224.0 * 1.5 * wing_area**0.5 * lambda_wing ** (-0.4) * thickness_term
        )
        partials[output_name, "data:geometry:wing:aspect_ratio"] = (
            224.0 * base * (-0.4) * lambda_wing ** (-1.0) * thickness_term
        )
        partials[output_name, "data:geometry:wing:root:thickness_ratio"] = 224.0 * base * 0.6
        partials[output_name, "data:geometry:wing:tip:thickness_ratio"] = 224.0 * base * 0.4
