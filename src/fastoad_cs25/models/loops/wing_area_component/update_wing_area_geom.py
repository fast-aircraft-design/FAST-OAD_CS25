"""
Computation of wing area following geometric constraints
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2023 ONERA & ISAE-SUPAERO
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
from fastoad.module_management.service_registry import RegisterSubmodel

from fastoad_cs25.models.loops.constants import (
    SERVICE_WING_AREA_CONSTRAINT_GEOM,
    SERVICE_WING_AREA_LOOP_GEOM,
)


@RegisterSubmodel(
    SERVICE_WING_AREA_LOOP_GEOM, "fastoad.submodel.loops.wing.area.update.geom.legacy"
)
class UpdateWingAreaGeom(om.ExplicitComponent):
    """Computes wing area for being able to load enough fuel to achieve the sizing mission."""

    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:weight:aircraft:sizing_block_fuel", val=np.nan, units="kg")

        # Same remark on the naming and connection as in the aero component
        self.add_output("wing_area:geom", val=100.0, units="m**2")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]
        mfw_mission = inputs["data:weight:aircraft:sizing_block_fuel"]

        avg_thickness_ratio = 0.6 * root_thickness_ratio + 0.4 * tip_thickness_ratio

        wing_area_mission = (
            max(1000.0, mfw_mission - 1570.0) / (224 * lambda_wing**-0.4 * avg_thickness_ratio)
        ) ** (1.0 / 1.5)

        outputs["wing_area:geom"] = wing_area_mission

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]
        mfw_mission = inputs["data:weight:aircraft:sizing_block_fuel"]

        avg_thickness_ratio = 0.6 * root_thickness_ratio + 0.4 * tip_thickness_ratio
        # Derivative of wing area wrt average thickness ratio
        d_wing_area_d_avg_toc = (
            -1.0
            / 1.5
            * (
                max(1000.0, mfw_mission - 1570.0)
                / (224.0 * lambda_wing**-0.4 * avg_thickness_ratio ** (5.0 / 2.0))
            )
            ** (1.0 / 1.5)
        )

        partials["wing_area:geom", "data:geometry:wing:aspect_ratio"] = (
            0.4
            / 1.5
            * (
                max(1000.0, mfw_mission - 1570.0)
                / (224 * avg_thickness_ratio * lambda_wing ** (11.0 / 10.0))
            )
            ** (1.0 / 1.5)
        )
        partials["wing_area:geom", "data:geometry:wing:root:thickness_ratio"] = (
            d_wing_area_d_avg_toc * 0.6
        )
        partials["wing_area:geom", "data:geometry:wing:tip:thickness_ratio"] = (
            d_wing_area_d_avg_toc * 0.4
        )
        if mfw_mission < 1000.0:
            partials["wing_area:geom", "data:weight:aircraft:sizing_block_fuel"] = 0.0
        else:
            partials["wing_area:geom", "data:weight:aircraft:sizing_block_fuel"] = (
                (1.0 / (224 * lambda_wing**-0.4 * avg_thickness_ratio)) ** (1.0 / 1.5)
                * (1.0 / 1.5)
                * (mfw_mission - 1570.0) ** (-1.0 / 3.0)
            )


@RegisterSubmodel(
    SERVICE_WING_AREA_CONSTRAINT_GEOM, "fastoad.submodel.loops.wing.area.constraint.geom.legacy"
)
class WingAreaConstraintsGeom(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:aircraft:sizing_block_fuel", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:additional_fuel_capacity", units="kg")

        # Value are easy to compute so they'll be given here
        self.declare_partials(
            of="data:weight:aircraft:additional_fuel_capacity",
            wrt="data:weight:aircraft:sizing_block_fuel",
            val=-1,
        )
        self.declare_partials(
            of="data:weight:aircraft:additional_fuel_capacity",
            wrt="data:weight:aircraft:MFW",
            val=1,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw = inputs["data:weight:aircraft:MFW"]
        mission_fuel = inputs["data:weight:aircraft:sizing_block_fuel"]

        outputs["data:weight:aircraft:additional_fuel_capacity"] = mfw - mission_fuel
