#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2026 ONERA & ISAE-SUPAERO
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

from ..constants import SERVICE_EMPTY_AIRCRAFT_CG_Y


@oad.RegisterSubmodel(
    SERVICE_EMPTY_AIRCRAFT_CG_Y, "fastoad.submodel.weight.cg.empty_aircraft.y.legacy"
)
class ComputeCGY(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "cg_y_item_names",
            default=[
                "data:weight:airframe:wing:",
                "data:weight:airframe:fuselage:",
                "data:weight:airframe:horizontal_tail:",
                "data:weight:airframe:vertical_tail:",
                "data:weight:airframe:flight_controls:",
                "data:weight:airframe:landing_gear:main:",
                "data:weight:airframe:landing_gear:front:",
                "data:weight:airframe:pylon:",
                "data:weight:airframe:paint:",
                "data:weight:propulsion:engine:",
                "data:weight:propulsion:fuel_lines:",
                "data:weight:propulsion:unconsumables:",
                "data:weight:systems:power:auxiliary_power_unit:",
                "data:weight:systems:power:electric_systems:",
                "data:weight:systems:power:hydraulic_systems:",
                "data:weight:systems:life_support:insulation:",
                "data:weight:systems:life_support:air_conditioning:",
                "data:weight:systems:life_support:de-icing:",
                "data:weight:systems:life_support:cabin_lighting:",
                "data:weight:systems:life_support:seats_crew_accommodation:",
                "data:weight:systems:life_support:oxygen:",
                "data:weight:systems:life_support:safety_equipment:",
                "data:weight:systems:navigation:",
                "data:weight:systems:transmission:",
                "data:weight:systems:operational:radar:",
                "data:weight:systems:operational:cargo_hold:",
                "data:weight:systems:flight_kit:",
                "data:weight:furniture:passenger_seats:",
                "data:weight:furniture:food_water:",
                "data:weight:furniture:security_kit:",
                "data:weight:furniture:toilets:",
            ],
            desc="Names of the items for consideration in the computation of the aircraft's empty "
            "CG in the y-axis. Items' names will be appended with 'CG:y' for the position of "
            "the item's CG and 'mass' for the item's mass.",
        )

    def setup(self):
        for item_names in self.options["cg_y_item_names"]:
            # By default, the aircraft is symmetrical but user inputs or other models can change it.
            self.add_input(item_names + "CG:y", val=0.0, units="m")
            self.add_input(item_names + "mass", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:CG:y", units="m")

    def setup_partials(self):
        for item_names in self.options["cg_y_item_names"]:
            self.declare_partials(
                "data:weight:aircraft_empty:CG:y", item_names + "mass", method="exact"
            )
            self.declare_partials(
                "data:weight:aircraft_empty:CG:y", item_names + "CG:y", method="exact"
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total_moment = 0.0
        total_mass = 0.0

        for item_names in self.options["cg_y_item_names"]:
            total_moment += inputs[item_names + "CG:y"] * inputs[item_names + "mass"]
            total_mass += inputs[item_names + "mass"]

        outputs["data:weight:aircraft_empty:CG:y"] = total_moment / total_mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        total_moment = 0.0
        total_mass = 0.0

        # Need to run it once to get the denominator and common terms
        for item_names in self.options["cg_y_item_names"]:
            total_mass += inputs[item_names + "mass"]
            total_moment += inputs[item_names + "CG:y"] * inputs[item_names + "mass"]

        for item_names in self.options["cg_y_item_names"]:
            partials["data:weight:aircraft_empty:CG:y", item_names + "CG:y"] = (
                inputs[item_names + "mass"] / total_mass
            )
            partials["data:weight:aircraft_empty:CG:y", item_names + "mass"] = (
                inputs[item_names + "CG:y"] * total_mass - total_moment
            ) / total_mass**2.0
