"""
Estimation of wing geometry
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
import openmdao.api as om

from .constants import (
    SERVICE_WING_GEOMETRY_PLANFORM,
    SERVICE_WING_GEOMETRY_THICKNESS,
    SERVICE_WING_GEOMETRY_WET_AREA,
    SERVICE_WING_GEOMETRY_MFW,
)
from ...constants import SERVICE_WING_GEOMETRY


@oad.RegisterSubmodel(SERVICE_WING_GEOMETRY, "fastoad.submodel.geometry.wing.legacy")
class ComputeWingGeometry(om.Group):
    """Wing geometry estimation"""

    def setup(self):
        self.add_subsystem(
            "y_wing",
            oad.RegisterSubmodel.get_submodel(SERVICE_WING_GEOMETRY_PLANFORM),
            promotes=["*"],
        )
        self.add_subsystem(
            "toc_wing",
            oad.RegisterSubmodel.get_submodel(SERVICE_WING_GEOMETRY_THICKNESS),
            promotes=["*"],
        )
        self.add_subsystem(
            "wetarea_wing",
            oad.RegisterSubmodel.get_submodel(SERVICE_WING_GEOMETRY_WET_AREA),
            promotes=["*"],
        )
        self.add_subsystem(
            "mfw", oad.RegisterSubmodel.get_submodel(SERVICE_WING_GEOMETRY_MFW), promotes=["*"]
        )
