#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025 ONERA & ISAE-SUPAERO
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

import openmdao.api as om

# from fastoad.module_management.constants import ModelDomain
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem, RegisterSubmodel

from .constants import (
    SERVICE_GUST_LOADS,
    SERVICE_MANEUVER_LOADS,
    SERVICE_SIZING_LOADS_ENVELOPE,
)


@RegisterOpenMDAOSystem("fastoad.loads.legacy")  # TODO add ModelDomain.LOADS
class ComputeLoads(om.Group):
    def initialize(self):
        self.options.declare(
            "fuel_load_alleviation",
            types=bool,
            default=True,
            desc="If False this simulates a dry wing,"
            "i.e. the sizing load 2 does not take into account the fuel weight.",
        )

    def setup(self):
        fuel_load_alleviation_option = {
            "fuel_load_alleviation": self.options["fuel_load_alleviation"]
        }
        self.add_subsystem(
            "gust_loads",
            RegisterSubmodel.get_submodel(SERVICE_GUST_LOADS, fuel_load_alleviation_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "maneuver_loads",
            RegisterSubmodel.get_submodel(SERVICE_MANEUVER_LOADS, fuel_load_alleviation_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "sizing_loads_envelope",
            RegisterSubmodel.get_submodel(SERVICE_SIZING_LOADS_ENVELOPE),
            promotes=["*"],
        )
