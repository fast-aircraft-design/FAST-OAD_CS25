"""Main components for mass breakdown."""
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
import openmdao.api as om
from fastoad.module_management.service_registry import RegisterSubmodel

from ...constants import PAYLOAD_FROM_NPAX
from ..constants import SERVICE_MASS_BREAKDOWN
from .constants import (
    SERVICE_AIRFRAME_MASS,
    SERVICE_CREW_MASS,
    SERVICE_FURNITURE_MASS,
    SERVICE_MLW_MZFW,
    SERVICE_OWE,
    SERVICE_PAYLOAD_MASS,
    SERVICE_PROPULSION_MASS,
    SERVICE_SYSTEMS_MASS,
)


@RegisterSubmodel(SERVICE_MASS_BREAKDOWN, "fastoad.submodel.weight.mass.legacy")
class MassBreakdown(
    oad.CycleGroup,
    default_linear_solver="om.LinearBlockGS",
    default_linear_options={"iprint": 0},
    default_nonlinear_options={"maxiter": 50, "iprint": 0},
):
    """
    Computes analytically the mass of each part of the aircraft, and the resulting sum,
    the Overall Weight Empty (OWE).

    Some models depend on MZFW (Max Zero Fuel Weight), MLW (Max Landing Weight) and
    MTOW (Max TakeOff Weight), which depend on OWE.

    This model cycles for having consistent OWE, MZFW and MLW.

    Options:
    - payload_from_npax: If True (default), payload masses will be computed from NPAX, if False
                         design payload mass and maximum payload mass must be provided.
    """

    def initialize(self):
        super().initialize()
        self.options.declare(PAYLOAD_FROM_NPAX, types=bool, default=True)

    def setup(self):
        super().setup()
        if self.options[PAYLOAD_FROM_NPAX]:
            self.add_subsystem(
                "payload",
                RegisterSubmodel.get_submodel(SERVICE_PAYLOAD_MASS),
                promotes=["*"],
            )
        self.add_subsystem("owe", RegisterSubmodel.get_submodel(SERVICE_OWE), promotes=["*"])
        self.add_subsystem(
            "update_mzfw_and_mlw", RegisterSubmodel.get_submodel(SERVICE_MLW_MZFW), promotes=["*"]
        )


@RegisterSubmodel(SERVICE_OWE, "fastoad.submodel.weight.mass.owe.legacy")
class OperatingWeightEmpty(om.Group):
    """Operating Empty Weight (OEW) estimation.

    This group aggregates weight from all components of the aircraft.
    """

    def setup(self):
        # Propulsion should be done before airframe, because it drives pylon mass.
        self.add_subsystem(
            "propulsion_weight",
            RegisterSubmodel.get_submodel(SERVICE_PROPULSION_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "airframe_weight",
            RegisterSubmodel.get_submodel(SERVICE_AIRFRAME_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "systems_weight",
            RegisterSubmodel.get_submodel(SERVICE_SYSTEMS_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "furniture_weight",
            RegisterSubmodel.get_submodel(SERVICE_FURNITURE_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "crew_weight",
            RegisterSubmodel.get_submodel(SERVICE_CREW_MASS),
            promotes=["*"],
        )

        weight_sum = om.AddSubtractComp()
        weight_sum.add_equation(
            "data:weight:aircraft:OWE",
            [
                "data:weight:airframe:mass",
                "data:weight:propulsion:mass",
                "data:weight:systems:mass",
                "data:weight:furniture:mass",
                "data:weight:crew:mass",
            ],
            units="kg",
            desc="Mass of crew",
        )

        self.add_subsystem("OWE_sum", weight_sum, promotes=["*"])
