"""
Estimation of power systems weight
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

from .constants import SERVICE_POWER_SYSTEMS_MASS


@oad.RegisterSubmodel(
    SERVICE_POWER_SYSTEMS_MASS, "fastoad.submodel.weight.mass.systems.power.legacy"
)
class PowerSystemsWeight(om.ExplicitComponent):
    """
    Weight estimation for power systems (generation and distribution)

    This includes:

    - Auxiliary Power Unit (APU)
    - electric systems
    - hydraulic systems

    Based on formulas in :cite:`supaero:2014`, mass contribution C1
    """

    def setup(self):
        self.add_input("data:geometry:cabin:NPAX1", val=np.nan, units="unitless")
        self.add_input("data:weight:airframe:flight_controls:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input(
            "tuning:weight:systems:power:auxiliary_power_unit:mass:k", val=1.0, units="unitless"
        )
        self.add_input(
            "tuning:weight:systems:power:auxiliary_power_unit:mass:offset", val=0.0, units="kg"
        )
        self.add_input(
            "tuning:weight:systems:power:electric_systems:mass:k", val=1.0, units="unitless"
        )
        self.add_input(
            "tuning:weight:systems:power:electric_systems:mass:offset", val=0.0, units="kg"
        )
        self.add_input(
            "tuning:weight:systems:power:hydraulic_systems:mass:k", val=1.0, units="unitless"
        )
        self.add_input(
            "tuning:weight:systems:power:hydraulic_systems:mass:offset", val=0.0, units="kg"
        )
        self.add_input("settings:weight:systems:power:mass:k_elec", val=1.0, units="unitless")

        self.add_output("data:weight:systems:power:auxiliary_power_unit:mass", units="kg")
        self.add_output("data:weight:systems:power:electric_systems:mass", units="kg")
        self.add_output("data:weight:systems:power:hydraulic_systems:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    # pylint: disable=too-many-locals
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        npax1 = inputs["data:geometry:cabin:NPAX1"]
        flight_controls_weight = inputs["data:weight:airframe:flight_controls:mass"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        k_c11 = inputs["tuning:weight:systems:power:auxiliary_power_unit:mass:k"]
        offset_c11 = inputs["tuning:weight:systems:power:auxiliary_power_unit:mass:offset"]
        k_c12 = inputs["tuning:weight:systems:power:electric_systems:mass:k"]
        offset_c12 = inputs["tuning:weight:systems:power:electric_systems:mass:offset"]
        k_c13 = inputs["tuning:weight:systems:power:hydraulic_systems:mass:k"]
        offset_c13 = inputs["tuning:weight:systems:power:hydraulic_systems:mass:offset"]
        k_elec = inputs["settings:weight:systems:power:mass:k_elec"]

        # Mass of auxiliary power unit
        temp_c11 = 11.3 * npax1**0.64
        outputs["data:weight:systems:power:auxiliary_power_unit:mass"] = (
            k_c11 * temp_c11 + offset_c11
        )

        # Mass of electric system
        temp_c12 = k_elec * (0.444 * mtow**0.66 + 2.54 * npax1 + 0.254 * flight_controls_weight)
        outputs["data:weight:systems:power:electric_systems:mass"] = k_c12 * temp_c12 + offset_c12

        # Mass of the hydraulic system
        temp_c13 = k_elec * (0.256 * mtow**0.66 + 1.46 * npax1 + 0.146 * flight_controls_weight)
        outputs["data:weight:systems:power:hydraulic_systems:mass"] = k_c13 * temp_c13 + offset_c13

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        npax1 = inputs["data:geometry:cabin:NPAX1"]
        flight_controls_weight = inputs["data:weight:airframe:flight_controls:mass"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        k_c11 = inputs["tuning:weight:systems:power:auxiliary_power_unit:mass:k"]
        k_c12 = inputs["tuning:weight:systems:power:electric_systems:mass:k"]
        k_c13 = inputs["tuning:weight:systems:power:hydraulic_systems:mass:k"]
        k_elec = inputs["settings:weight:systems:power:mass:k_elec"]

        temp_c11 = 11.3 * npax1**0.64
        electric_term = 0.444 * mtow**0.66 + 2.54 * npax1 + 0.254 * flight_controls_weight
        hydraulic_term = 0.256 * mtow**0.66 + 1.46 * npax1 + 0.146 * flight_controls_weight
        temp_c12 = k_elec * electric_term
        temp_c13 = k_elec * hydraulic_term

        apu = "data:weight:systems:power:auxiliary_power_unit:mass"
        electric = "data:weight:systems:power:electric_systems:mass"
        hydraulic = "data:weight:systems:power:hydraulic_systems:mass"

        partials[apu, "data:geometry:cabin:NPAX1"] = k_c11 * 11.3 * 0.64 * npax1 ** (-0.36)
        partials[apu, "tuning:weight:systems:power:auxiliary_power_unit:mass:k"] = temp_c11
        partials[apu, "tuning:weight:systems:power:auxiliary_power_unit:mass:offset"] = 1.0

        partials[electric, "data:weight:aircraft:MTOW"] = (
            k_c12 * k_elec * 0.444 * 0.66 * mtow ** (-0.34)
        )
        partials[electric, "data:geometry:cabin:NPAX1"] = k_c12 * k_elec * 2.54
        partials[electric, "data:weight:airframe:flight_controls:mass"] = k_c12 * k_elec * 0.254
        partials[electric, "settings:weight:systems:power:mass:k_elec"] = k_c12 * electric_term
        partials[electric, "tuning:weight:systems:power:electric_systems:mass:k"] = temp_c12
        partials[electric, "tuning:weight:systems:power:electric_systems:mass:offset"] = 1.0

        partials[hydraulic, "data:weight:aircraft:MTOW"] = (
            k_c13 * k_elec * 0.256 * 0.66 * mtow ** (-0.34)
        )
        partials[hydraulic, "data:geometry:cabin:NPAX1"] = k_c13 * k_elec * 1.46
        partials[hydraulic, "data:weight:airframe:flight_controls:mass"] = k_c13 * k_elec * 0.146
        partials[hydraulic, "settings:weight:systems:power:mass:k_elec"] = k_c13 * hydraulic_term
        partials[hydraulic, "tuning:weight:systems:power:hydraulic_systems:mass:k"] = temp_c13
        partials[hydraulic, "tuning:weight:systems:power:hydraulic_systems:mass:offset"] = 1.0
