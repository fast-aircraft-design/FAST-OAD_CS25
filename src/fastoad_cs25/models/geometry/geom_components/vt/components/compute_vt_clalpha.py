"""
Estimation of vertical tail lift coefficient
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

# TODO: This belongs more to aerodynamics than geometry


class ComputeVTClalpha(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail lift coefficient estimation"""

    def setup(self):
        self.add_input("data:TLAR:cruise_mach", val=np.nan, units="unitless")
        self.add_input("data:geometry:has_T_tail", val=np.nan, units="unitless")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan, units="unitless")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")

        self.add_output("data:aerodynamics:vertical_tail:high_speed:CL_alpha", units="1/rad")

    def setup_partials(self):
        self.declare_partials(
            "data:aerodynamics:vertical_tail:high_speed:CL_alpha", "*", method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tail_type = np.round(inputs["data:geometry:has_T_tail"])
        cruise_mach = inputs["data:TLAR:cruise_mach"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        k_ar_effective = 2.9 if tail_type == 1 else 1.55
        lambda_vt = inputs["data:geometry:vertical_tail:aspect_ratio"] * k_ar_effective

        beta = np.sqrt(1 - cruise_mach**2)
        cl_alpha_vt = (
            0.8
            * 2
            * np.pi
            * lambda_vt
            / (
                2
                + np.sqrt(
                    4
                    + lambda_vt**2
                    * beta**2
                    / 0.95**2
                    * (1 + (np.tan(sweep_25_vt / 180.0 * np.pi)) ** 2 / beta**2)
                )
            )
        )

        outputs["data:aerodynamics:vertical_tail:high_speed:CL_alpha"] = cl_alpha_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tail_type = np.round(inputs["data:geometry:has_T_tail"])
        cruise_mach = inputs["data:TLAR:cruise_mach"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        k_ar_effective = 2.9 if tail_type == 1 else 1.55
        lambda_vt = inputs["data:geometry:vertical_tail:aspect_ratio"] * k_ar_effective

        beta2 = 1.0 - cruise_mach**2
        sweep_rad = sweep_25_vt / 180.0 * np.pi
        tan_sweep = np.tan(sweep_rad)
        root_term = 4.0 + lambda_vt**2 / 0.9025 * (beta2 + tan_sweep**2)
        sqrt_term = np.sqrt(root_term)
        denominator = 2.0 + sqrt_term
        numerator = 0.8 * 2.0 * np.pi * lambda_vt

        d_root_d_lambda = 2.0 * lambda_vt / 0.9025 * (beta2 + tan_sweep**2)
        d_root_d_mach = -2.0 * cruise_mach * lambda_vt**2 / 0.9025
        d_root_d_sweep = (
            lambda_vt**2 / 0.9025 * 2.0 * tan_sweep / np.cos(sweep_rad) ** 2 * np.pi / 180.0
        )

        output_name = "data:aerodynamics:vertical_tail:high_speed:CL_alpha"
        partials[output_name, "data:geometry:vertical_tail:aspect_ratio"] = (
            0.8 * 2.0 * np.pi * k_ar_effective / denominator
            - numerator / denominator**2 * 0.5 / sqrt_term * d_root_d_lambda * k_ar_effective
        )
        partials[output_name, "data:TLAR:cruise_mach"] = (
            -numerator / denominator**2 * 0.5 / sqrt_term * d_root_d_mach
        )
        partials[output_name, "data:geometry:vertical_tail:sweep_25"] = (
            -numerator / denominator**2 * 0.5 / sqrt_term * d_root_d_sweep
        )
