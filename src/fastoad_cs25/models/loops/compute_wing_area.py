"""
Computation of wing area
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

import numpy as np
import openmdao.api as om
from fastoad.module_management.constants import ModelDomain
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem, RegisterSubmodel

from .constants import (
    SERVICE_WING_AREA_CONSTRAINT_AERO,
    SERVICE_WING_AREA_CONSTRAINT_GEOM,
    SERVICE_WING_AREA_LOOP_AERO,
    SERVICE_WING_AREA_LOOP_GEOM,
)


@RegisterOpenMDAOSystem("fastoad.loop.wing_area", domain=ModelDomain.OTHER)
class ComputeWingArea(om.Group):
    """
    Computes needed wing area for:
      - having enough lift at required approach speed
      - being able to load enough fuel to achieve the sizing mission
    """

    def initialize(self):
        super().initialize()
        self.options.declare("use_fuel", default=True, types=bool)
        self.options.declare("use_approach_speed", default=True, types=bool)

    def setup(self):
        if self.options["use_fuel"]:
            self.add_subsystem(
                "wing_area_geom",
                RegisterSubmodel.get_submodel(SERVICE_WING_AREA_LOOP_GEOM),
                promotes_inputs=["*"],
                promotes_outputs=[],
            )
            self.connect("wing_area_geom.wing_area:geom", "wing_area.wing_area:geom")

        if self.options["use_approach_speed"]:
            self.add_subsystem(
                "wing_area_aero",
                RegisterSubmodel.get_submodel(SERVICE_WING_AREA_LOOP_AERO),
                promotes_inputs=["*"],
                promotes_outputs=[],
            )
            self.connect("wing_area_aero.wing_area:aero", "wing_area.wing_area:aero")

        if self.options["use_fuel"] or self.options["use_approach_speed"]:
            self.add_subsystem(
                "wing_area",
                _ComputeWingArea(),
                promotes_inputs=[],
                promotes_outputs=["*"],
            )
        self.add_subsystem(
            "geom_constraint",
            RegisterSubmodel.get_submodel(SERVICE_WING_AREA_CONSTRAINT_GEOM),
            promotes=["*"],
        )
        self.add_subsystem(
            "aero_constraint",
            RegisterSubmodel.get_submodel(SERVICE_WING_AREA_CONSTRAINT_AERO),
            promotes=["*"],
        )


class _ComputeWingArea(om.ExplicitComponent):
    """
    Computation of wing area from needed approach speed and mission fuel and taking whichever
    is greatest
    """

    def setup(self):
        self.add_input("wing_area:aero", units="m**2", val=0.0)
        self.add_input("wing_area:geom", units="m**2", val=0.0)

        self.add_output("data:geometry:wing:area", val=100.0, units="m**2")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area_mission = inputs["wing_area:geom"]
        wing_area_approach = inputs["wing_area:aero"]

        outputs["data:geometry:wing:area"] = np.nanmax([wing_area_mission, wing_area_approach])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_area_mission = inputs["wing_area:geom"]
        wing_area_approach = inputs["wing_area:aero"]

        partials["data:geometry:wing:area", "wing_area:geom"] = (
            1.0 if wing_area_mission > wing_area_approach else 0.0
        )
        partials["data:geometry:wing:area", "wing_area:aero"] = (
            0.0 if wing_area_mission > wing_area_approach else 1.0
        )
