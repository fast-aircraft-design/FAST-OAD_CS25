"""
Submodel for computing wing planform.
"""
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

from ..constants import SERVICE_WING_GEOMETRY_PLANFORM
from .compute_b_50 import ComputeB50
from .compute_center_chord import ComputeCenterChord
from .compute_l1_l4 import ComputeL1AndL4Wing
from .compute_l2_l3 import ComputeL2AndL3Wing
from .compute_mac_wing import ComputeMACWing
from .compute_relative_kink import ComputeRelativeKink
from .compute_sweep_wing import ComputeInnerSweepWing, ComputeSweepWing
from .compute_x_wing import ComputeXWing
from .compute_y_wing import ComputeYWing


@oad.RegisterSubmodel(
    SERVICE_WING_GEOMETRY_PLANFORM, "fastoad.submodel.geometry.wing.planform.legacy"
)
class ComputeWingGeometry(
    oad.CycleGroup,
    default_linear_options={"iprint": 0},
    default_nonlinear_options={"maxiter": 50, "iprint": 0},
):
    """Computation of wing planform"""

    def initialize(self):
        super().initialize()
        self.options.declare("impose_sweep_100_inner", types=bool, default=False)
        self.options.declare("impose_absolute_kink", types=bool, default=False)

    def setup(self):
        super().setup()
        self.add_subsystem(
            "y_wing",
            ComputeYWing(impose_absolute_kink=self.options["impose_absolute_kink"]),
            promotes=["*"],
        )
        self.add_subsystem("l14_wing", ComputeL1AndL4Wing(), promotes=["*"])
        self.add_subsystem("l2l3_wing", ComputeL2AndL3Wing(), promotes=["*"])
        self.add_subsystem("x_wing", ComputeXWing(), promotes=["*"])
        self.add_subsystem("mac_wing", ComputeMACWing(), promotes=["*"])
        self.add_subsystem("b50_wing", ComputeB50(), promotes=["*"])
        self.add_subsystem("sweep_wing", ComputeSweepWing(), promotes=["*"])
        self.add_subsystem("compute_center_chord", ComputeCenterChord(), promotes=["*"])
        if not self.options["impose_sweep_100_inner"]:
            self.add_subsystem("eval_sweep_inner", ComputeInnerSweepWing(), promotes=["*"])
        if self.options["impose_absolute_kink"]:
            self.add_subsystem("eval_relative_kink", ComputeRelativeKink(), promotes=["*"])
