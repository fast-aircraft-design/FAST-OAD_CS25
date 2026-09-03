"""
Estimation centers of gravity of other components
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

from ..constants import SERVICE_OTHERS_CG


@oad.RegisterSubmodel(SERVICE_OTHERS_CG, "fastoad.submodel.weight.cg.others.legacy")
class ComputeOthersCG(om.Group):
    def setup(self):
        self.add_subsystem(name="compute_cg_x_others", subsys=ComputeOthersCGX(), promotes=["*"])
        self.add_subsystem(name="compute_cg_z_others", subsys=ComputeOthersCGZ(), promotes=["*"])


class ComputeOthersCGX(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Other components center of gravities estimation"""

    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:furniture:passenger_seats:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:geometry:cabin:NPAX1", val=np.nan, units="unitless")
        self.add_input(
            "data:geometry:cabin:seats:economical:count_by_row", val=np.nan, units="unitless"
        )
        self.add_input("data:geometry:cabin:seats:economical:length", val=np.nan, units="m")
        self.add_input(
            "settings:weight:airframe:landing_gear:front:CG:position_ratio_on_front_fuselage",
            val=0.75,
            units="unitless",
        )

        # TODO: add description of these CGs
        self.add_output("data:weight:airframe:fuselage:CG:x", units="m")
        self.add_output("data:weight:airframe:landing_gear:front:CG:x", units="m")
        self.add_output("data:weight:airframe:pylon:CG:x", units="m")
        self.add_output("data:weight:airframe:paint:CG:x", units="m")
        self.add_output("data:weight:propulsion:fuel_lines:CG:x", units="m")
        self.add_output("data:weight:propulsion:unconsumables:CG:x", units="m")
        self.add_output("data:weight:systems:power:auxiliary_power_unit:CG:x", units="m")
        self.add_output("data:weight:systems:power:electric_systems:CG:x", units="m")
        self.add_output("data:weight:systems:power:hydraulic_systems:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:insulation:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:air_conditioning:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:de-icing:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:cabin_lighting:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:seats_crew_accommodation:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:oxygen:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:safety_equipment:CG:x", units="m")
        self.add_output("data:weight:systems:navigation:CG:x", units="m")
        self.add_output("data:weight:systems:transmission:CG:x", units="m")
        self.add_output("data:weight:systems:operational:radar:CG:x", units="m")
        self.add_output("data:weight:systems:operational:cargo_hold:CG:x", units="m")
        self.add_output("data:weight:furniture:food_water:CG:x", units="m")
        self.add_output("data:weight:furniture:security_kit:CG:x", units="m")
        self.add_output("data:weight:furniture:toilets:CG:x", units="m")
        self.add_output("data:weight:payload:PAX:CG:x", units="m")
        self.add_output("data:weight:payload:rear_fret:CG:x", units="m")
        self.add_output("data:weight:payload:front_fret:CG:x", units="m")

    def setup_partials(self):
        self.declare_partials(
            "data:weight:airframe:fuselage:CG:x", "data:geometry:fuselage:length", method="fd"
        )
        self.declare_partials(
            "data:weight:airframe:landing_gear:front:CG:x",
            [
                "data:geometry:fuselage:front_length",
                "settings:weight:airframe:landing_gear:front:CG:position_ratio_on_front_fuselage",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:weight:airframe:pylon:CG:x", "data:weight:propulsion:engine:CG:x", method="fd"
        )
        self.declare_partials(
            "data:weight:propulsion:fuel_lines:CG:x",
            "data:weight:propulsion:engine:CG:x",
            method="fd",
        )
        self.declare_partials(
            "data:weight:propulsion:unconsumables:CG:x",
            "data:weight:propulsion:engine:CG:x",
            method="fd",
        )
        self.declare_partials(
            [
                "data:weight:systems:power:auxiliary_power_unit:CG:x",
                "data:weight:systems:power:electric_systems:CG:x",
                "data:weight:systems:power:hydraulic_systems:CG:x",
                "data:weight:systems:life_support:insulation:CG:x",
                "data:weight:systems:life_support:cabin_lighting:CG:x",
                "data:weight:systems:transmission:CG:x",
                "data:weight:systems:operational:radar:CG:x",
            ],
            "data:geometry:fuselage:length",
            method="fd",
        )
        self.declare_partials(
            [
                "data:weight:systems:life_support:air_conditioning:CG:x",
                "data:weight:systems:life_support:seats_crew_accommodation:CG:x",
                "data:weight:systems:life_support:oxygen:CG:x",
                "data:weight:systems:operational:cargo_hold:CG:x",
            ],
            "data:weight:furniture:passenger_seats:CG:x",
            method="fd",
        )
        self.declare_partials(
            "data:weight:systems:life_support:de-icing:CG:x",
            ["data:geometry:wing:MAC:at25percent:x", "data:geometry:wing:MAC:length"],
            method="fd",
        )
        self.declare_partials(
            "data:weight:systems:life_support:safety_equipment:CG:x",
            [
                "data:weight:propulsion:engine:mass",
                "data:weight:propulsion:engine:CG:x",
                "data:geometry:cabin:NPAX1",
                "data:weight:furniture:passenger_seats:CG:x",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:weight:systems:navigation:CG:x",
            "data:geometry:fuselage:front_length",
            method="fd",
        )
        self.declare_partials(
            "data:weight:furniture:food_water:CG:x",
            [
                "data:geometry:fuselage:length",
                "data:geometry:fuselage:rear_length",
                "data:geometry:cabin:seats:economical:length",
                "data:geometry:cabin:seats:economical:count_by_row",
            ],
            method="fd",
        )
        self.declare_partials(
            ["data:weight:furniture:security_kit:CG:x", "data:weight:furniture:toilets:CG:x"],
            "data:weight:furniture:passenger_seats:CG:x",
            method="fd",
        )
        self.declare_partials(
            "data:weight:payload:PAX:CG:x",
            "data:weight:furniture:passenger_seats:CG:x",
            method="fd",
        )
        self.declare_partials(
            "data:weight:payload:rear_fret:CG:x",
            [
                "data:geometry:fuselage:rear_length",
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:leading_edge:x:local",
                "data:geometry:wing:root:chord",
                "data:geometry:cabin:seats:economical:count_by_row",
                "data:geometry:cabin:seats:economical:length",
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:fuselage:length",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:weight:payload:front_fret:CG:x",
            [
                "data:geometry:fuselage:front_length",
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:leading_edge:x:local",
                "data:geometry:wing:MAC:at25percent:x",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs):
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        fus_length = inputs["data:geometry:fuselage:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        x_cg_b1 = inputs["data:weight:propulsion:engine:CG:x"]
        x_cg_d2 = inputs["data:weight:furniture:passenger_seats:CG:x"]
        weight_engines = inputs["data:weight:propulsion:engine:mass"]
        npax1 = inputs["data:geometry:cabin:NPAX1"]
        front_seat_number_eco = inputs["data:geometry:cabin:seats:economical:count_by_row"]
        ls_eco = inputs["data:geometry:cabin:seats:economical:length"]
        front_lg_position = inputs[
            "settings:weight:airframe:landing_gear:front:CG:position_ratio_on_front_fuselage"
        ]

        # Assume fuselage CG is at 45% of fuselage length
        x_cg_a2 = 0.45 * fus_length

        x_cg_a52 = lav * front_lg_position
        x_cg_a6 = x_cg_b1
        x_cg_b2 = x_cg_b1
        x_cg_b3 = x_cg_b1

        # APU is installed after the pressure bulkhead, and pressurized area is
        # about 80% of fuselage length
        x_cg_c11 = 0.95 * fus_length
        x_cg_c12 = 0.5 * fus_length
        x_cg_c13 = 0.5 * fus_length
        x_cg_c21 = 0.45 * fus_length
        x_cg_c22 = x_cg_d2
        x_cg_c23 = fa_length - 0.15 * l0_wing
        x_cg_c24 = 0.45 * fus_length
        x_cg_c25 = x_cg_d2
        x_cg_c26 = x_cg_d2
        x_cg_c27 = (0.01 * weight_engines * x_cg_b1 + 2.3 * npax1 * x_cg_d2) / (
            0.01 * weight_engines + 2.3 * npax1
        )
        x_cg_c3 = lav * 0.8
        x_cg_c4 = 0.5 * fus_length
        x_cg_c51 = 0.02 * fus_length
        x_cg_c52 = x_cg_d2

        x_cg_d3 = lav + (fus_length - lav - lar) + ls_eco * front_seat_number_eco + 0.92 + 0.432
        x_cg_d5 = x_cg_d2

        length_front_fret = fa_length - 0.25 * l0_wing - x0_wing - lav
        x_cg_front_fret = lav + length_front_fret * 0.5

        length_rear_fret = (
            fus_length
            - lar
            + (front_seat_number_eco - 5) * ls_eco
            - (lav + length_front_fret + 0.8 * l2_wing)
        )

        x_cg_rear_fret = lav + length_front_fret + 0.8 * l2_wing + length_rear_fret * 0.5

        x_cg_pl = x_cg_d2

        outputs["data:weight:airframe:fuselage:CG:x"] = x_cg_a2
        outputs["data:weight:airframe:landing_gear:front:CG:x"] = x_cg_a52
        outputs["data:weight:airframe:pylon:CG:x"] = x_cg_a6
        outputs["data:weight:airframe:paint:CG:x"] = 0.0

        outputs["data:weight:propulsion:fuel_lines:CG:x"] = x_cg_b2
        outputs["data:weight:propulsion:unconsumables:CG:x"] = x_cg_b3

        outputs["data:weight:systems:power:auxiliary_power_unit:CG:x"] = x_cg_c11
        outputs["data:weight:systems:power:electric_systems:CG:x"] = x_cg_c12
        outputs["data:weight:systems:power:hydraulic_systems:CG:x"] = x_cg_c13
        outputs["data:weight:systems:life_support:insulation:CG:x"] = x_cg_c21
        outputs["data:weight:systems:life_support:air_conditioning:CG:x"] = x_cg_c22
        outputs["data:weight:systems:life_support:de-icing:CG:x"] = x_cg_c23
        outputs["data:weight:systems:life_support:cabin_lighting:CG:x"] = x_cg_c24
        outputs["data:weight:systems:life_support:seats_crew_accommodation:CG:x"] = x_cg_c25
        outputs["data:weight:systems:life_support:oxygen:CG:x"] = x_cg_c26
        outputs["data:weight:systems:life_support:safety_equipment:CG:x"] = x_cg_c27
        outputs["data:weight:systems:navigation:CG:x"] = x_cg_c3
        outputs["data:weight:systems:transmission:CG:x"] = x_cg_c4
        outputs["data:weight:systems:operational:radar:CG:x"] = x_cg_c51
        outputs["data:weight:systems:operational:cargo_hold:CG:x"] = x_cg_c52

        outputs["data:weight:furniture:food_water:CG:x"] = x_cg_d3
        outputs["data:weight:furniture:security_kit:CG:x"] = x_cg_d5
        outputs["data:weight:furniture:toilets:CG:x"] = x_cg_d5
        outputs["data:weight:payload:PAX:CG:x"] = x_cg_pl
        outputs["data:weight:payload:rear_fret:CG:x"] = x_cg_rear_fret
        outputs["data:weight:payload:front_fret:CG:x"] = x_cg_front_fret


class ComputeOthersCGZ(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Other components center of gravities estimation"""

    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:dihedral", val=6.0 * np.pi / 180.0, units="rad")
        self.add_input("data:geometry:propulsion:pylon:height", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:airframe:wing:CG:z", val=np.nan, units="m")

        self.add_output("data:weight:airframe:fuselage:CG:z", units="m")
        self.add_output("data:weight:airframe:landing_gear:front:CG:z", units="m")
        self.add_output("data:weight:airframe:landing_gear:main:CG:z", units="m")
        self.add_output("data:weight:airframe:pylon:CG:z", units="m")
        self.add_output("data:weight:airframe:paint:CG:z", units="m")
        self.add_output("data:weight:propulsion:fuel_lines:CG:z", units="m")
        self.add_output("data:weight:propulsion:unconsumables:CG:z", units="m")
        self.add_output("data:weight:systems:power:auxiliary_power_unit:CG:z", units="m")
        self.add_output("data:weight:systems:power:electric_systems:CG:z", units="m")
        self.add_output("data:weight:systems:power:hydraulic_systems:CG:z", units="m")
        self.add_output("data:weight:systems:life_support:insulation:CG:z", units="m")
        self.add_output("data:weight:systems:life_support:air_conditioning:CG:z", units="m")
        self.add_output("data:weight:systems:life_support:de-icing:CG:z", units="m")
        self.add_output("data:weight:systems:life_support:cabin_lighting:CG:z", units="m")
        self.add_output("data:weight:systems:life_support:seats_crew_accommodation:CG:z", units="m")
        self.add_output("data:weight:systems:life_support:oxygen:CG:z", units="m")
        self.add_output("data:weight:systems:life_support:safety_equipment:CG:z", units="m")
        self.add_output("data:weight:systems:navigation:CG:z", units="m")
        self.add_output("data:weight:systems:transmission:CG:z", units="m")
        self.add_output("data:weight:systems:operational:radar:CG:z", units="m")
        self.add_output("data:weight:systems:operational:cargo_hold:CG:z", units="m")
        self.add_output("data:weight:furniture:food_water:CG:z", units="m")
        self.add_output("data:weight:furniture:security_kit:CG:z", units="m")
        self.add_output("data:weight:furniture:toilets:CG:z", units="m")
        self.add_output("data:weight:payload:PAX:CG:z", units="m")
        self.add_output("data:weight:payload:rear_fret:CG:z", units="m")
        self.add_output("data:weight:payload:front_fret:CG:z", units="m")

    def setup_partials(self):
        self.declare_partials(
            of=[
                "data:weight:airframe:fuselage:CG:z",
                "data:weight:airframe:paint:CG:z",
                "data:weight:systems:life_support:insulation:CG:z",
                "data:weight:systems:operational:radar:CG:z",
            ],
            wrt="data:geometry:fuselage:maximum_height",
            val=0.5,
        )
        self.declare_partials(
            of="data:weight:airframe:pylon:CG:z",
            wrt=[
                "data:geometry:propulsion:nacelle:y",
                "data:geometry:wing:dihedral",
                "data:geometry:propulsion:pylon:height",
            ],
            method="exact",
        )
        self.declare_partials(
            of=[
                "data:weight:propulsion:fuel_lines:CG:z",
                "data:weight:propulsion:unconsumables:CG:z",
            ],
            wrt="data:weight:propulsion:engine:CG:z",
            val=1.0,
        )
        self.declare_partials(
            of=[
                "data:weight:systems:power:auxiliary_power_unit:CG:z",
                "data:weight:systems:life_support:cabin_lighting:CG:z",
                "data:weight:systems:life_support:seats_crew_accommodation:CG:z",
                "data:weight:systems:life_support:oxygen:CG:z",
                "data:weight:systems:life_support:safety_equipment:CG:z",
                "data:weight:systems:navigation:CG:z",
                "data:weight:systems:transmission:CG:z",
                "data:weight:furniture:food_water:CG:z",
                "data:weight:furniture:security_kit:CG:z",
                "data:weight:furniture:toilets:CG:z",
            ],
            wrt="data:geometry:fuselage:maximum_height",
            val=2.0 / 3.0,
        )
        self.declare_partials(
            of=[
                "data:weight:systems:power:electric_systems:CG:z",
                "data:weight:systems:power:hydraulic_systems:CG:z",
                "data:weight:systems:life_support:air_conditioning:CG:z",
                "data:weight:systems:operational:cargo_hold:CG:z",
            ],
            wrt="data:geometry:fuselage:maximum_height",
            val=1.0 / 6.0,
        )
        self.declare_partials(
            of="data:weight:systems:life_support:de-icing:CG:z",
            wrt="data:weight:airframe:wing:CG:z",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuselage_height = inputs["data:geometry:fuselage:maximum_height"]
        dihedral = inputs["data:geometry:wing:dihedral"]
        engine_y = inputs["data:geometry:propulsion:nacelle:y"]
        pylon_height = inputs["data:geometry:propulsion:pylon:height"]
        z_cg_b1 = inputs["data:weight:propulsion:engine:CG:z"]
        z_cg_a1 = inputs["data:weight:airframe:wing:CG:z"]

        # For the CG in the Z direction of many components, we will make a simple assumption that
        # the fuselage is divided in two part, under the floor which occupy the bottom third of
        # the fuselage height and the cabin which occupy the remaining two thirds

        outputs["data:weight:airframe:fuselage:CG:z"] = fuselage_height / 2.0
        outputs["data:weight:airframe:landing_gear:front:CG:z"] = 0.0
        outputs["data:weight:airframe:landing_gear:main:CG:z"] = 0.0
        outputs["data:weight:airframe:pylon:CG:z"] = (
            engine_y * np.sin(dihedral) - pylon_height / 2.0
        )
        outputs["data:weight:airframe:paint:CG:z"] = fuselage_height / 2.0

        outputs["data:weight:propulsion:fuel_lines:CG:z"] = z_cg_b1
        outputs["data:weight:propulsion:unconsumables:CG:z"] = z_cg_b1

        # APU is flush with the top of the fuselage
        outputs["data:weight:systems:power:auxiliary_power_unit:CG:z"] = 2.0 * fuselage_height / 3.0
        outputs["data:weight:systems:power:electric_systems:CG:z"] = fuselage_height / 6.0
        outputs["data:weight:systems:power:hydraulic_systems:CG:z"] = fuselage_height / 6.0
        outputs["data:weight:systems:life_support:insulation:CG:z"] = fuselage_height / 2.0
        outputs["data:weight:systems:life_support:air_conditioning:CG:z"] = fuselage_height / 6.0
        outputs["data:weight:systems:life_support:de-icing:CG:z"] = z_cg_a1
        outputs["data:weight:systems:life_support:cabin_lighting:CG:z"] = (
            2.0 * fuselage_height / 3.0
        )
        outputs["data:weight:systems:life_support:seats_crew_accommodation:CG:z"] = (
            2.0 * fuselage_height / 3.0
        )
        outputs["data:weight:systems:life_support:oxygen:CG:z"] = 2.0 * fuselage_height / 3.0
        outputs["data:weight:systems:life_support:safety_equipment:CG:z"] = (
            2.0 * fuselage_height / 3.0
        )
        outputs["data:weight:systems:navigation:CG:z"] = 2.0 * fuselage_height / 3.0
        outputs["data:weight:systems:transmission:CG:z"] = 2.0 * fuselage_height / 3.0
        outputs["data:weight:systems:operational:radar:CG:z"] = fuselage_height / 2.0
        outputs["data:weight:systems:operational:cargo_hold:CG:z"] = fuselage_height / 6.0

        outputs["data:weight:furniture:food_water:CG:z"] = 2.0 * fuselage_height / 3.0
        outputs["data:weight:furniture:security_kit:CG:z"] = 2.0 * fuselage_height / 3.0
        outputs["data:weight:furniture:toilets:CG:z"] = 2.0 * fuselage_height / 3.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dihedral = inputs["data:geometry:wing:dihedral"]
        engine_y = inputs["data:geometry:propulsion:nacelle:y"]

        partials["data:weight:airframe:pylon:CG:z", "data:geometry:wing:dihedral"] = (
            engine_y * np.cos(dihedral)
        )
        partials["data:weight:airframe:pylon:CG:z", "data:geometry:propulsion:nacelle:y"] = np.sin(
            dihedral
        )
        partials["data:weight:airframe:pylon:CG:z", "data:geometry:propulsion:pylon:height"] = -0.5
