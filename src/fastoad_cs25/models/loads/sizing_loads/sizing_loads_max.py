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

        self.add_output("data:mission:sizing:cs25:sizing_load", units="N")

    def setup_partials(self):
        self.declare_partials(
            "data:mission:sizing:cs25:sizing_load",
            [
                "data:mission:sizing:cs25:envelope:max_sizing_load_1",
                "data:mission:sizing:cs25:envelope:max_sizing_load_2",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n1m1 = inputs["data:mission:sizing:cs25:envelope:max_sizing_load_1"]
        n2m2 = inputs["data:mission:sizing:cs25:envelope:max_sizing_load_2"]

        n_m = max(n1m1, n2m2)

        outputs["data:mission:sizing:cs25:sizing_load"] = n_m
