"""
FAST - Copyright (c) 2016 ONERA ISAE
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
from fastoad.module_management.service_registry import RegisterSubmodel

from .constants import (
    SERVICE_AIRCRAFT_CG,
    SERVICE_FLIGHT_CONTROLS_CG,
    SERVICE_GLOBAL_CG,
    SERVICE_HORIZONTAL_TAIL_CG,
    SERVICE_MLG_CG,
    SERVICE_OTHERS_CG,
    SERVICE_TANKS_CG,
    SERVICE_VERTICAL_TAIL_CG,
    SERVICE_WING_CG,
)
from ..constants import SERVICE_CENTERS_OF_GRAVITY


@RegisterSubmodel(SERVICE_CENTERS_OF_GRAVITY, "fastoad.submodel.weight.cg.legacy")
class CG(
    oad.CycleGroup,
    default_linear_solver="om.LinearBlockGS",
    default_linear_options={"iprint": 0},
    default_nonlinear_options={"maxiter": 200, "iprint": 0},
):
    """Model that computes the global center of gravity"""

    def setup(self):
        super().setup()
        self.add_subsystem(
            "ht_cg", RegisterSubmodel.get_submodel(SERVICE_HORIZONTAL_TAIL_CG), promotes=["*"]
        )
        self.add_subsystem(
            "vt_cg", RegisterSubmodel.get_submodel(SERVICE_VERTICAL_TAIL_CG), promotes=["*"]
        )
        self.add_subsystem(
            "compute_cg_wing", RegisterSubmodel.get_submodel(SERVICE_WING_CG), promotes=["*"]
        )
        self.add_subsystem(
            "compute_cg_control_surface",
            RegisterSubmodel.get_submodel(SERVICE_FLIGHT_CONTROLS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_cg_tanks", RegisterSubmodel.get_submodel(SERVICE_TANKS_CG), promotes=["*"]
        )
        self.add_subsystem(
            "compute_cg_others", RegisterSubmodel.get_submodel(SERVICE_OTHERS_CG), promotes=["*"]
        )
        self.add_subsystem(
            "compute_cg", RegisterSubmodel.get_submodel(SERVICE_GLOBAL_CG), promotes=["*"]
        )
        self.add_subsystem(
            "update_mlg", RegisterSubmodel.get_submodel(SERVICE_MLG_CG), promotes=["*"]
        )
        self.add_subsystem(
            "aircraft", RegisterSubmodel.get_submodel(SERVICE_AIRCRAFT_CG), promotes=["*"]
        )


@RegisterSubmodel(SERVICE_AIRCRAFT_CG, "fastoad.submodel.weight.cg.aircraft.legacy")
class ComputeAircraftCG(om.ExplicitComponent):
    """Compute position of aircraft CG from CG ratio"""

    def setup(self):
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:CG:aft:x", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        cg_ratio = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        mac_position = inputs["data:geometry:wing:MAC:at25percent:x"]

        outputs["data:weight:aircraft:CG:aft:x"] = (
            mac_position - 0.25 * l0_wing + cg_ratio * l0_wing
        )
