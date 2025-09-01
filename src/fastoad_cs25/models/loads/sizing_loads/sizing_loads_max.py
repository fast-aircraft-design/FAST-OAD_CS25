"""
Python module for the computation of the maximum sizing loads.
"""
#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
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

from ..constants import SERVICE_SIZING_LOADS_MAX


@oad.RegisterSubmodel(SERVICE_SIZING_LOADS_MAX, "fastoad.submodel.loads.maximum_sizing.legacy")
class SizingLoadsEnvelope(om.ExplicitComponent):
    """
    Computes CS25 sizing loading

    """

    def setup(self):
        self.add_input("data:mission:sizing:cs25:envelope:max_sizing_load_1", units="N")
        self.add_input("data:mission:sizing:cs25:envelope:max_sizing_load_2", units="N")
        self.add_input("data:mission:sizing:cs25:envelope:max_load_factor_1", units="unitless")
        self.add_input("data:mission:sizing:cs25:envelope:max_load_factor_2", units="unitless")

        self.add_output("data:mission:sizing:cs25:sizing_load", units="N")
        self.add_output("data:mission:sizing:cs25:sizing_load_factor", units="unitless")

    def setup_partials(self):
        self.declare_partials(
            "data:mission:sizing:cs25:sizing_load",
            "*",
            method="fd",
        )

        self.declare_partials(
            "data:mission:sizing:cs25:sizing_load_factor",
            "*",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n1m1 = inputs["data:mission:sizing:cs25:envelope:max_sizing_load_1"]
        n2m2 = inputs["data:mission:sizing:cs25:envelope:max_sizing_load_2"]
        n1 = inputs["data:mission:sizing:cs25:envelope:max_load_factor_1"]
        n2 = inputs["data:mission:sizing:cs25:envelope:max_load_factor_2"]

        id_max = int(np.argmax([n1m1, n2m2]))
        nm = [n1m1, n2m2][id_max]
        n = [n1, n2][id_max]

        outputs["data:mission:sizing:cs25:sizing_load"] = nm
        outputs["data:mission:sizing:cs25:sizing_load_factor"] = n
