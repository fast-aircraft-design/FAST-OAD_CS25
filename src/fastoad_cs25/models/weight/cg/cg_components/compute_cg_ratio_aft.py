"""
Estimation of center of gravity ratio with aft
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from ..constants import SERVICE_EMPTY_AIRCRAFT_CG_X


@oad.RegisterSubmodel(
    SERVICE_EMPTY_AIRCRAFT_CG_X, "fastoad.submodel.weight.cg.empty_aircraft.x.legacy"
)
class ComputeCGXRatioAft(om.Group):
    def setup(self):
        self.add_subsystem("cg_x_all", ComputeCGX(), promotes=["*"])
        self.add_subsystem("cg_x_ratio", CGXRatio(), promotes=["*"])


class ComputeCGX(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "cg_x_item_names",
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
            "CG in the x-axis. Items' names will be appended with 'CG:x' for the position of "
            "the item's CG and 'mass' for the item's mass.",
        )

    def setup(self):
        for item_names in self.options["cg_x_item_names"]:
            self.add_input(item_names + "CG:x", val=np.nan, units="m")
            self.add_input(item_names + "mass", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:mass", units="kg")
        self.add_output("data:weight:aircraft_empty:CG:x", units="m")

    def setup_partials(self):
        for item_names in self.options["cg_x_item_names"]:
            self.declare_partials("data:weight:aircraft_empty:mass", item_names + "mass", val=1.0)
            self.declare_partials(
                "data:weight:aircraft_empty:CG:x", item_names + "mass", method="exact"
            )
            self.declare_partials(
                "data:weight:aircraft_empty:CG:x", item_names + "CG:x", method="exact"
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total_moment = 0.0
        total_mass = 0.0

        for item_names in self.options["cg_x_item_names"]:
            total_moment += inputs[item_names + "CG:x"] * inputs[item_names + "mass"]
            total_mass += inputs[item_names + "mass"]

        outputs["data:weight:aircraft_empty:mass"] = total_mass
        outputs["data:weight:aircraft_empty:CG:x"] = total_moment / total_mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        total_moment = 0.0
        total_mass = 0.0

        # Need to run it once to get the denominator and common terms
        for item_names in self.options["cg_x_item_names"]:
            total_mass += inputs[item_names + "mass"]
            total_moment += inputs[item_names + "CG:x"] * inputs[item_names + "mass"]

        for item_names in self.options["cg_x_item_names"]:
            partials["data:weight:aircraft_empty:CG:x", item_names + "CG:x"] = (
                inputs[item_names + "mass"] / total_mass
            )
            partials["data:weight:aircraft_empty:CG:x", item_names + "mass"] = (
                inputs[item_names + "CG:x"] * total_mass - total_moment
            ) / total_mass**2.0


class CGXRatio(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:empty:CG:MAC_position", units="unitless")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x_cg_all = inputs["data:weight:aircraft_empty:CG:x"]
        wing_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        mac = inputs["data:geometry:wing:MAC:length"]

        outputs["data:weight:aircraft:empty:CG:MAC_position"] = (
            x_cg_all - wing_position + 0.25 * mac
        ) / mac

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x_cg_all = inputs["data:weight:aircraft_empty:CG:x"]
        wing_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        mac = inputs["data:geometry:wing:MAC:length"]

        partials[
            "data:weight:aircraft:empty:CG:MAC_position", "data:weight:aircraft_empty:CG:x"
        ] = 1.0 / mac
        partials[
            "data:weight:aircraft:empty:CG:MAC_position", "data:geometry:wing:MAC:at25percent:x"
        ] = -1.0 / mac
        partials["data:weight:aircraft:empty:CG:MAC_position", "data:geometry:wing:MAC:length"] = (
            -(x_cg_all - wing_position) / mac**2.0
        )
