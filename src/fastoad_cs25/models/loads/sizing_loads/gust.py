"""
Python module for the computation of gust sizing load cases.
"""
#  This file is part of FAST-OAD_CS25
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from scipy.constants import g
from stdatm import Atmosphere

from ..constants import SERVICE_GUST_LOADS


@oad.RegisterSubmodel(SERVICE_GUST_LOADS, "fastoad.submodel.loads.gust.legacy")
class GustLoads(om.ExplicitComponent):
    """
    Computes CS25 vertical gust load factors evaluated at two different load cases:

    Load case 1: with wings with almost no fuel
    Load case 2: at maximum take-off weight

    Based on formulas in :cite:`supaero:2014`, §6.3

    """

    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:cruise:CL_alpha", val=np.nan, units="1/rad")
        self.add_input("data:load_case:lc1:U_gust", val=np.nan, units="m/s")
        self.add_input("data:load_case:lc1:altitude", val=np.nan, units="ft")
        self.add_input("data:load_case:lc1:Vc_EAS", val=np.nan, units="m/s")
        self.add_input("data:load_case:lc2:U_gust", val=np.nan, units="m/s")
        self.add_input("data:load_case:lc2:altitude", val=np.nan, units="ft")
        self.add_input("data:load_case:lc2:Vc_EAS", val=np.nan, units="m/s")
        self.add_input("data:load_case:gust_intensity", val=1.0, units="unitless")
        self.add_input("data:mission:sizing:cs25:safety_factor", val=1.5, units="unitless")

        self.add_output("data:mission:sizing:cs25:gust:load_factor_1", units="unitless")
        self.add_output("data:mission:sizing:cs25:gust:load_factor_2", units="unitless")

    def setup_partials(self):
        self.declare_partials(
            "data:mission:sizing:cs25:gust:load_factor_1",
            [
                "data:geometry:wing:area",
                "data:geometry:wing:span",
                "data:weight:aircraft:MZFW",
                "data:aerodynamics:aircraft:cruise:CL_alpha",
                "data:load_case:lc1:U_gust",
                "data:load_case:lc1:altitude",
                "data:load_case:lc1:Vc_EAS",
                "data:load_case:gust_intensity",
                "data:mission:sizing:cs25:safety_factor",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:mission:sizing:cs25:gust:load_factor_2",
            [
                "data:geometry:wing:area",
                "data:geometry:wing:span",
                "data:weight:aircraft:MTOW",
                "data:aerodynamics:aircraft:cruise:CL_alpha",
                "data:load_case:lc2:U_gust",
                "data:load_case:lc2:altitude",
                "data:load_case:lc2:Vc_EAS",
                "data:load_case:gust_intensity",
                "data:mission:sizing:cs25:safety_factor",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sea_level_density = Atmosphere(0).density
        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]
        mzfw = inputs["data:weight:aircraft:MZFW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        cl_alpha = inputs["data:aerodynamics:aircraft:cruise:CL_alpha"]
        u_gust1 = inputs["data:load_case:lc1:U_gust"]
        alt_1 = inputs["data:load_case:lc1:altitude"]
        vc_eas1 = inputs["data:load_case:lc1:Vc_EAS"]
        u_gust2 = inputs["data:load_case:lc2:U_gust"]
        alt_2 = inputs["data:load_case:lc2:altitude"]
        vc_eas2 = inputs["data:load_case:lc2:Vc_EAS"]
        gust_intensity = inputs["data:load_case:gust_intensity"]
        safety_factor = inputs["data:mission:sizing:cs25:safety_factor"]

        # calculation of mean geometric chord
        chord_geom = wing_area / span

        # load case #1
        m1 = 1.05 * mzfw
        n_gust_1 = self.__n_gust(
            m1,
            wing_area,
            Atmosphere(alt_1).density,
            sea_level_density,
            chord_geom,
            vc_eas1,
            cl_alpha,
            u_gust1,
        )
        n1 = safety_factor * n_gust_1 * gust_intensity

        # load case #2
        n_gust_2 = self.__n_gust(
            mtow,
            wing_area,
            Atmosphere(alt_2).density,
            sea_level_density,
            chord_geom,
            vc_eas2,
            cl_alpha,
            u_gust2,
        )
        n2 = safety_factor * n_gust_2 * gust_intensity

        outputs["data:mission:sizing:cs25:gust:load_factor_1"] = n1
        outputs["data:mission:sizing:cs25:gust:load_factor_2"] = n2

    @staticmethod
    def __n_gust(mass, wing_area, rho, sea_level_density, chord_geom, vc_eas, cl_alpha, u_gust):
        """
        Computes a reference vertical gust load factor.

        :param mass: Aircraft mass [kg]
        :param wing_area: Wing reference area [m2]
        :param rho: Air density at flight altitude [kg/m3]
        :param sea_level_density: Air density at sea level [kg/m3]
        :param chord_geom: Geometric mean aerodynamic chord [m]
        :param vc_eas: Equivalent airspeed at cruising condition Vc [m/s]
        :param cl_alpha: Wing lift alpha curve slope [1/rad]
        :param u_gust: Gust vertical velocity [m/s]
        :return: Gust load factor (n_gust) [dimensionless]
        """
        mu_g = 2 * mass / rho / wing_area / chord_geom / cl_alpha
        k_g = 0.88 * mu_g / (5.3 + mu_g)  # attenuation factor
        return 1 + (sea_level_density / 2 / g) * k_g * u_gust * (
            vc_eas * cl_alpha / (mass / wing_area)
        )  # n_gust
