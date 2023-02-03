"""Computation of CL characteristics at low speed."""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2023 ONERA & ISAE-SUPAERO
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
from fastoad.module_management.service_registry import RegisterSubmodel

from ..constants import SERVICE_ALPHA


@RegisterSubmodel(SERVICE_ALPHA, "fastoad.submodel.aerodynamics.alpha.legacy")
class ComputeAlpha(om.ExplicitComponent):
    """
    Computes a linear evolution of angle of attack from CL scale and CL gradient.

    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        polar_type = "low_speed" if self.options["low_speed_aero"] else "cruise"
        CL0_default = 0.2 if self.options["low_speed_aero"] else 0.1  # pylint: disable=invalid-name
        self.add_input(
            f"data:aerodynamics:aircraft:{polar_type}:CL", shape_by_conn=True, val=np.nan
        )
        self.add_input(
            f"data:aerodynamics:aircraft:{polar_type}:CL_alpha", val=np.nan, units="1/rad"
        )
        self.add_input(f"data:aerodynamics:aircraft:{polar_type}:CL0", val=CL0_default)

        self.add_output(
            f"data:aerodynamics:aircraft:{polar_type}:AoA",
            copy_shape=f"data:aerodynamics:aircraft:{polar_type}:CL",
            units="rad",
        )

    def setup_partials(self):
        polar_type = "low_speed" if self.options["low_speed_aero"] else "cruise"
        self.declare_partials(f"data:aerodynamics:aircraft:{polar_type}:AoA", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        polar_type = "low_speed" if self.options["low_speed_aero"] else "cruise"
        CL = inputs[f"data:aerodynamics:aircraft:{polar_type}:CL"]  # pylint: disable=invalid-name
        CLalpha = inputs[  # pylint: disable=invalid-name
            f"data:aerodynamics:aircraft:{polar_type}:CL_alpha"
        ]
        CL0 = inputs[f"data:aerodynamics:aircraft:{polar_type}:CL0"]  # pylint: disable=invalid-name

        outputs[f"data:aerodynamics:aircraft:{polar_type}:AoA"] = (CL - CL0) / CLalpha
