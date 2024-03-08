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
import openmdao.api as om

from .compute_b_50 import ComputeB50
from .compute_l1_l4 import ComputeL1AndL4Wing
from .compute_l2_l3 import ComputeL2AndL3Wing
from .compute_mac_wing import ComputeMACWing
from .compute_sweep_wing import ComputeSweepWing
from .compute_x_wing import ComputeXWing
from .compute_y_wing import ComputeYWing
from ..constants import SERVICE_WING_GEOMETRY_PLANFORM


@oad.RegisterSubmodel(
    SERVICE_WING_GEOMETRY_PLANFORM, "fastoad.submodel.geometry.wing.planform.legacy"
)
class ComputeWingGeometry(om.Group):
    """Computation of wing planform"""

    def setup(self):
        # These solvers are needed because sweep angle of inner trailing edge depends
        # on sweep angle of outer trailing edge, that depends on (virtual) root chord,
        # that depend on sweep angle of inner trailing edge.
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.DirectSolver()

        self.add_subsystem("y_wing", ComputeYWing(), promotes=["*"])
        self.add_subsystem("l14_wing", ComputeL1AndL4Wing(), promotes=["*"])
        self.add_subsystem("l2l3_wing", ComputeL2AndL3Wing(), promotes=["*"])
        self.add_subsystem("x_wing", ComputeXWing(), promotes=["*"])
        self.add_subsystem("mac_wing", ComputeMACWing(), promotes=["*"])
        self.add_subsystem("b50_wing", ComputeB50(), promotes=["*"])
        self.add_subsystem("sweep_wing", ComputeSweepWing(), promotes=["*"])
