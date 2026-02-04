"""Computation of form drag for whole aircraft."""
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

import fastoad.api as oad
import openmdao.api as om

from ..constants import (
    SERVICE_CD0,
    SERVICE_CD0_FUSELAGE,
    SERVICE_CD0_HORIZONTAL_TAIL,
    SERVICE_CD0_NACELLES_PYLONS,
    SERVICE_CD0_SUM,
    SERVICE_CD0_VERTICAL_TAIL,
    SERVICE_CD0_WING,
)


@oad.RegisterSubmodel(SERVICE_CD0, "fastoad.submodel.aerodynamics.CD0.legacy")
class CD0(om.Group):
    """
    Computation of form drag for whole aircraft.

    Computes and sums the drag coefficients of all components.
    Interaction drag is assumed to be taken into account at component level.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        low_speed_option = {"low_speed_aero": self.options["low_speed_aero"]}
        self.add_subsystem(
            "cd0_wing",
            oad.RegisterSubmodel.get_submodel(SERVICE_CD0_WING, low_speed_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cd0_fuselage",
            oad.RegisterSubmodel.get_submodel(SERVICE_CD0_FUSELAGE, low_speed_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cd0_ht",
            oad.RegisterSubmodel.get_submodel(SERVICE_CD0_HORIZONTAL_TAIL, low_speed_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cd0_vt",
            oad.RegisterSubmodel.get_submodel(SERVICE_CD0_VERTICAL_TAIL, low_speed_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cd0_nac_pylons",
            oad.RegisterSubmodel.get_submodel(SERVICE_CD0_NACELLES_PYLONS, low_speed_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cd0_total",
            oad.RegisterSubmodel.get_submodel(SERVICE_CD0_SUM, low_speed_option),
            promotes=["*"],
        )
