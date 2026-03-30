"""Computation of CL and CD for whole aircraft."""
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
import numpy as np
import openmdao.api as om

from ..constants import SERVICE_POLAR, PolarType


@oad.RegisterSubmodel(SERVICE_POLAR, "fastoad.submodel.aerodynamics.polar.legacy")
class ComputePolar(om.ExplicitComponent):
    """Computation of CL and CD for whole aircraft."""

    def initialize(self):
        self.options.declare("polar_type", default=PolarType.HIGH_SPEED, types=PolarType)

    def setup(self):
        self.add_input("tuning:aerodynamics:aircraft:high_speed:CD:k", val=np.nan)
        self.add_input("tuning:aerodynamics:aircraft:high_speed:CD:offset", val=np.nan)
        self.add_input("tuning:aerodynamics:aircraft:high_speed:CD:winglet_effect:k", val=np.nan)
        self.add_input(
            "tuning:aerodynamics:aircraft:high_speed:CD:winglet_effect:offset", val=np.nan
        )

        if self.options["polar_type"] != PolarType.HIGH_SPEED:
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:CD:CD0",
                shape_by_conn=True,
                val=np.nan,
            )
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:CL",
                shape_by_conn=True,
                val=np.nan,
            )
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:CD:trim",
                shape_by_conn=True,
                val=np.nan,
            )
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:induced_drag_coefficient",
                val=np.nan,
            )

            if self.options["polar_type"] == PolarType.TAKEOFF:
                self.add_input("data:aerodynamics:high_lift_devices:takeoff:CL", val=np.nan)
                self.add_input("data:aerodynamics:high_lift_devices:takeoff:CD", val=np.nan)
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CL",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CD",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CD:CD0",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CD:induced",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CD:wave",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CD:trim",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output("data:aerodynamics:aircraft:takeoff:CD:viscous")

            elif self.options["polar_type"] == PolarType.LANDING:
                self.add_input("data:aerodynamics:high_lift_devices:landing:CL", val=np.nan)
                self.add_input("data:aerodynamics:high_lift_devices:landing:CD", val=np.nan)
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CL",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CD",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CD:CD0",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CD:induced",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CD:wave",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CD:trim",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output("data:aerodynamics:aircraft:landing:CD:viscous")

            elif self.options["polar_type"] == PolarType.LOW_SPEED:
                self.add_output(
                    "data:aerodynamics:aircraft:low_speed:CD",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:low_speed:CD:induced",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:low_speed:CD:wave",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output("data:aerodynamics:aircraft:low_speed:CD:viscous")

            else:
                raise AttributeError(f"Unknown polar type: {self.options['polar_type']}")

        elif self.options["polar_type"] == PolarType.HIGH_SPEED:
            self.add_input(
                "data:aerodynamics:aircraft:high_speed:CL", shape_by_conn=True, val=np.nan
            )
            self.add_input(
                "data:aerodynamics:aircraft:high_speed:CD:CD0",
                shape_by_conn=True,
                val=np.nan,
            )
            self.add_input(
                "data:aerodynamics:aircraft:high_speed:CD:trim",
                shape_by_conn=True,
                val=np.nan,
            )
            self.add_input(
                "data:aerodynamics:aircraft:high_speed:CD:wave",
                shape_by_conn=True,
                val=np.nan,
            )
            self.add_input(
                "data:aerodynamics:aircraft:high_speed:induced_drag_coefficient", val=np.nan
            )

            self.add_output(
                "data:aerodynamics:aircraft:high_speed:CD",
                copy_shape="data:aerodynamics:aircraft:high_speed:CL",
            )
            self.add_output(
                "data:aerodynamics:aircraft:high_speed:CD:induced",
                copy_shape="data:aerodynamics:aircraft:high_speed:CL",
            )
            self.add_output("data:aerodynamics:aircraft:high_speed:CD:viscous")
            self.add_output("data:aerodynamics:aircraft:high_speed:L_D_max")
            self.add_output("data:aerodynamics:aircraft:high_speed:optimal_CL")
            self.add_output("data:aerodynamics:aircraft:high_speed:optimal_CD")

        else:
            raise AttributeError(f"Unknown polar type: {self.options['polar_type']}")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        k_cd = inputs["tuning:aerodynamics:aircraft:high_speed:CD:k"]
        offset_cd = inputs["tuning:aerodynamics:aircraft:high_speed:CD:offset"]
        k_winglet_cd = inputs["tuning:aerodynamics:aircraft:high_speed:CD:winglet_effect:k"]
        offset_winglet_cd = inputs[
            "tuning:aerodynamics:aircraft:high_speed:CD:winglet_effect:offset"
        ]
        if self.options["polar_type"] != PolarType.HIGH_SPEED:
            cl = inputs["data:aerodynamics:aircraft:low_speed:CL"]
            cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD:CD0"]
            cd_trim = inputs["data:aerodynamics:aircraft:low_speed:CD:trim"]
            cd_c = 0.0
            coef_k = inputs["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
            if self.options["polar_type"] == PolarType.TAKEOFF:
                delta_cl_hl = inputs["data:aerodynamics:high_lift_devices:takeoff:CL"]
                delta_cd_hl = inputs["data:aerodynamics:high_lift_devices:takeoff:CD"]
            elif self.options["polar_type"] == PolarType.LANDING:
                delta_cl_hl = inputs["data:aerodynamics:high_lift_devices:landing:CL"]
                delta_cd_hl = inputs["data:aerodynamics:high_lift_devices:landing:CD"]
            elif self.options["polar_type"] == PolarType.LOW_SPEED:
                delta_cl_hl = 0.0
                delta_cd_hl = 0.0
            else:
                raise AttributeError(f"Unknown polar type: {self.options['polar_type']}")

        elif self.options["polar_type"] == PolarType.HIGH_SPEED:
            cl = inputs["data:aerodynamics:aircraft:high_speed:CL"]
            cd0 = inputs["data:aerodynamics:aircraft:high_speed:CD:CD0"]
            cd_trim = inputs["data:aerodynamics:aircraft:high_speed:CD:trim"]
            cd_c = inputs["data:aerodynamics:aircraft:high_speed:CD:wave"]
            coef_k = inputs["data:aerodynamics:aircraft:high_speed:induced_drag_coefficient"]
            delta_cl_hl = 0.0
            delta_cd_hl = 0.0
        else:
            raise AttributeError(f"Unknown polar type: {self.options['polar_type']}")

        cl = cl + delta_cl_hl
        cd_cd0 = cd0 * k_cd
        cd_wave = cd_c * k_cd
        cd_trim_component = cd_trim * k_cd
        cd_induced = coef_k * cl**2 * k_winglet_cd * k_cd
        cd_viscous = (offset_winglet_cd + delta_cd_hl) * k_cd + offset_cd
        cd = cd_cd0 + cd_wave + cd_trim_component + cd_induced + cd_viscous

        if self.options["polar_type"] == PolarType.LOW_SPEED:
            outputs["data:aerodynamics:aircraft:low_speed:CD"] = cd
            outputs["data:aerodynamics:aircraft:low_speed:CD:induced"] = cd_induced
            outputs["data:aerodynamics:aircraft:low_speed:CD:wave"] = cd_wave
            outputs["data:aerodynamics:aircraft:low_speed:CD:viscous"] = cd_viscous
        elif self.options["polar_type"] == PolarType.TAKEOFF:
            outputs["data:aerodynamics:aircraft:takeoff:CL"] = cl
            outputs["data:aerodynamics:aircraft:takeoff:CD"] = cd
            outputs["data:aerodynamics:aircraft:takeoff:CD:CD0"] = cd_cd0
            outputs["data:aerodynamics:aircraft:takeoff:CD:induced"] = cd_induced
            outputs["data:aerodynamics:aircraft:takeoff:CD:wave"] = cd_wave
            outputs["data:aerodynamics:aircraft:takeoff:CD:trim"] = cd_trim_component
            outputs["data:aerodynamics:aircraft:takeoff:CD:viscous"] = cd_viscous
        elif self.options["polar_type"] == PolarType.LANDING:
            outputs["data:aerodynamics:aircraft:landing:CL"] = cl
            outputs["data:aerodynamics:aircraft:landing:CD"] = cd
            outputs["data:aerodynamics:aircraft:landing:CD:CD0"] = cd_cd0
            outputs["data:aerodynamics:aircraft:landing:CD:induced"] = cd_induced
            outputs["data:aerodynamics:aircraft:landing:CD:wave"] = cd_wave
            outputs["data:aerodynamics:aircraft:landing:CD:trim"] = cd_trim_component
            outputs["data:aerodynamics:aircraft:landing:CD:viscous"] = cd_viscous
        else:
            outputs["data:aerodynamics:aircraft:high_speed:CD"] = cd
            outputs["data:aerodynamics:aircraft:high_speed:CD:induced"] = cd_induced
            outputs["data:aerodynamics:aircraft:high_speed:CD:viscous"] = cd_viscous

            Cl_opt, Cd_opt = get_optimum_ClCd(np.array([cd, cl]))[0:2]
            outputs["data:aerodynamics:aircraft:high_speed:L_D_max"] = Cl_opt / Cd_opt
            outputs["data:aerodynamics:aircraft:high_speed:optimal_CL"] = Cl_opt
            outputs["data:aerodynamics:aircraft:high_speed:optimal_CD"] = Cd_opt


def get_optimum_ClCd(ClCd):
    lift_drag_ratio = ClCd[1, :] / ClCd[0, :]
    optimum_index = np.argmax(lift_drag_ratio)

    optimum_Cl = ClCd[1][optimum_index]
    optimum_Cd = ClCd[0][optimum_index]

    return optimum_Cl, optimum_Cd
