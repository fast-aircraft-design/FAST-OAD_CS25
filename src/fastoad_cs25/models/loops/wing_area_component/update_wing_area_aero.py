"""
Computation of wing area following aerodynamic constraints
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
from scipy.constants import g

import openmdao.api as om

from fastoad.module_management.service_registry import RegisterSubmodel
from fastoad_cs25.models.loops.constants import (
    SERVICE_WING_AREA_LOOP_AERO,
    SERVICE_WING_AREA_CONSTRAINT_AERO,
)

from stdatm import Atmosphere


@RegisterSubmodel(
    SERVICE_WING_AREA_LOOP_AERO, "fastoad.submodel.loops.wing.area.update.aero.legacy"
)
class UpdateWingAreaAero(om.ExplicitComponent):
    """Computes wing area for having enough lift at required approach speed."""

    def setup(self):

        self.add_input("data:TLAR:approach_speed", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)

        # My plan is to not promote this variable and connect it by hand so that it foes not
        # appear in the output file as is but only as a constraints in the other component. We
        # could however give it a proper name.
        self.add_output("wing_area:aero", val=100.0, units="m**2")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        approach_speed = inputs["data:TLAR:approach_speed"]
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]

        rho_sl = Atmosphere(0.0).density
        stall_speed = approach_speed / 1.23

        wing_area_approach = mlw * g / (0.5 * rho_sl * stall_speed**2.0 * max_cl)

        outputs["wing_area:aero"] = wing_area_approach

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        approach_speed = inputs["data:TLAR:approach_speed"]
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]

        rho_sl = Atmosphere(0.0).density
        stall_speed = approach_speed / 1.23

        partials["wing_area:aero", "data:TLAR:approach_speed"] = (
            -2.0 * (mlw * g) / (0.5 * rho_sl * stall_speed**3.0 * max_cl) / 1.23
        )
        partials["wing_area:aero", "data:weight:aircraft:MLW"] = g / (
            0.5 * rho_sl * stall_speed**2.0 * max_cl
        )
        partials["wing_area:aero", "data:aerodynamics:aircraft:landing:CL_max"] = -(mlw * g) / (
            0.5 * rho_sl * stall_speed**2.0 * max_cl**2.0
        )


@RegisterSubmodel(
    SERVICE_WING_AREA_CONSTRAINT_AERO, "fastoad.submodel.loops.wing.area.constraint.aero.legacy"
)
class WingAreaConstraintsAero(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:TLAR:approach_speed", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:aerodynamics:aircraft:landing:additional_CL_capacity")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        approach_speed = inputs["data:TLAR:approach_speed"]
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        wing_area = inputs["data:geometry:wing:area"]

        rho_sl = Atmosphere(0.0).density
        stall_speed = approach_speed / 1.23

        outputs["data:aerodynamics:aircraft:landing:additional_CL_capacity"] = max_cl - mlw * g / (
            0.5 * rho_sl * stall_speed**2 * wing_area
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        approach_speed = inputs["data:TLAR:approach_speed"]
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = inputs["data:geometry:wing:area"]

        rho_sl = Atmosphere(0.0).density
        stall_speed = approach_speed / 1.23

        partials[
            "data:aerodynamics:aircraft:landing:additional_CL_capacity", "data:TLAR:approach_speed"
        ] = (2.0 * (mlw * g) / (0.5 * rho_sl * stall_speed**3.0 * wing_area) / 1.23)
        partials[
            "data:aerodynamics:aircraft:landing:additional_CL_capacity",
            "data:aerodynamics:aircraft:landing:CL_max",
        ] = 1.0
        partials[
            "data:aerodynamics:aircraft:landing:additional_CL_capacity",
            "data:weight:aircraft:MLW",
        ] = -g / (0.5 * rho_sl * stall_speed**2 * wing_area)
        partials[
            "data:aerodynamics:aircraft:landing:additional_CL_capacity",
            "data:geometry:wing:area",
        ] = (mlw * g) / (0.5 * rho_sl * stall_speed**2 * wing_area**2.0)
