"""
    Estimation of geometry of horizontal tail
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
import openmdao.api as om

from .components import ComputeHTChord, ComputeHTClalpha, ComputeHTMAC, ComputeHTSweep
from ...constants import SERVICE_HORIZONTAL_TAIL_GEOMETRY


@oad.RegisterSubmodel(
    SERVICE_HORIZONTAL_TAIL_GEOMETRY, "fastoad.submodel.geometry.horizontal_tail.legacy"
)
class ComputeHorizontalTailGeometry(om.Group):
    """Horizontal tail geometry estimation"""

    def setup(self):
        self.add_subsystem("ht_chord", ComputeHTChord(), promotes=["*"])
        self.add_subsystem("ht_mac", ComputeHTMAC(), promotes=["*"])
        self.add_subsystem("ht_sweep", ComputeHTSweep(), promotes=["*"])
        self.add_subsystem("ht_cl_alpha", ComputeHTClalpha(), promotes=["*"])
