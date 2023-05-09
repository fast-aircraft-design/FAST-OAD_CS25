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

from ..constants import SERVICE_CL_ALPHA


@RegisterSubmodel(SERVICE_CL_ALPHA, "fastoad.submodel.aerodynamics.CLalpha.legacy")
class ComputeCLAlpha(om.ExplicitComponent):
    """
    Computes CL gradient.

    CL gradient from :cite:`raymer:1999` Eq 12.6
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        if not self.options["low_speed_aero"]:
            self.add_input("data:TLAR:cruise_mach", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:aircraft:low_speed:CL_alpha", units="1/rad")
        else:
            self.add_output("data:aerodynamics:aircraft:cruise:CL_alpha", units="1/rad")

    def setup_partials(self):
        if self.options["low_speed_aero"]:
            self.declare_partials("data:aerodynamics:aircraft:low_speed:CL_alpha", "*", method="fd")
        else:
            self.declare_partials("data:aerodynamics:aircraft:cruise:CL_alpha", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        span = inputs["data:geometry:wing:span"]
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        wing_area = inputs["data:geometry:wing:area"]

        if self.options["low_speed_aero"]:
            mach = 0.2
        else:
            mach = inputs["data:TLAR:cruise_mach"]

        beta = np.sqrt(1.0 - mach**2)
        d_f = np.sqrt(width_max * height_max)
        fuselage_lift_factor = 1.07 * (1.0 + d_f / span) ** 2
        lambda_wing_eff = lambda_wing * (1.0 + 1.9 * tip_chord * tip_thickness_ratio / span)
        cl_alpha_wing = (
            2.0
            * np.pi
            * lambda_wing_eff
            / (
                2.0
                + np.sqrt(
                    4.0
                    + lambda_wing_eff**2
                    * beta**2
                    / 0.9025  # equals 0.95**2
                    * (1.0 + (np.tan(sweep_25)) ** 2 / beta**2)
                )
            )
            * (wing_area - root_chord * width_max)
            / wing_area
            * fuselage_lift_factor
        )

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:CL_alpha"] = cl_alpha_wing
        else:
            outputs["data:aerodynamics:aircraft:cruise:CL_alpha"] = cl_alpha_wing
