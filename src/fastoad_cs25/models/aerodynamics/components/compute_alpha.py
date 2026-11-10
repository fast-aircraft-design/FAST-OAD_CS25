"""Computation of CL characteristics at low speed."""
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

from ..constants import SERVICE_ALPHA


@oad.RegisterSubmodel(SERVICE_ALPHA, "fastoad.submodel.aerodynamics.alpha.legacy")
class ComputeAlpha(om.ExplicitComponent):
    """
    Computes a linear evolution of angle of attack from CL scale and CL gradient.

    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        polar_type = "low_speed" if self.options["low_speed_aero"] else "high_speed"
        CL0_default = 0.2 if self.options["low_speed_aero"] else 0.1  # pylint: disable=invalid-name
        self.add_input(
            f"data:aerodynamics:aircraft:{polar_type}:CL",
            shape_by_conn=True,
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            f"data:aerodynamics:aircraft:{polar_type}:CL_alpha",
            val=np.nan,
            units="1/rad",
        )
        self.add_input(
            f"data:aerodynamics:aircraft:{polar_type}:CL0", val=CL0_default, units="unitless"
        )

        self.add_output(
            f"data:aerodynamics:aircraft:{polar_type}:AoA",
            copy_shape=f"data:aerodynamics:aircraft:{polar_type}:CL",
            units="rad",
        )

    def setup_partials(self):
        polar_type = "low_speed" if self.options["low_speed_aero"] else "high_speed"
        self.declare_partials(f"data:aerodynamics:aircraft:{polar_type}:AoA", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        polar_type = "low_speed" if self.options["low_speed_aero"] else "high_speed"
        CL = inputs[f"data:aerodynamics:aircraft:{polar_type}:CL"]  # pylint: disable=invalid-name
        CLalpha = inputs[  # pylint: disable=invalid-name
            f"data:aerodynamics:aircraft:{polar_type}:CL_alpha"
        ]
        CL0 = inputs[f"data:aerodynamics:aircraft:{polar_type}:CL0"]  # pylint: disable=invalid-name

        outputs[f"data:aerodynamics:aircraft:{polar_type}:AoA"] = (CL - CL0) / CLalpha

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        polar_type = "low_speed" if self.options["low_speed_aero"] else "high_speed"
        output_name = f"data:aerodynamics:aircraft:{polar_type}:AoA"
        cl = inputs[f"data:aerodynamics:aircraft:{polar_type}:CL"]
        cl_alpha = inputs[f"data:aerodynamics:aircraft:{polar_type}:CL_alpha"]
        cl0 = inputs[f"data:aerodynamics:aircraft:{polar_type}:CL0"]

        size = cl.size
        partials[output_name, f"data:aerodynamics:aircraft:{polar_type}:CL"] = (
            np.eye(size) / cl_alpha
        )
        partials[output_name, f"data:aerodynamics:aircraft:{polar_type}:CL_alpha"] = (
            -(cl - cl0) / cl_alpha**2
        ).reshape((size, 1))
        partials[output_name, f"data:aerodynamics:aircraft:{polar_type}:CL0"] = np.full(
            (size, 1), -1.0 / cl_alpha
        )
