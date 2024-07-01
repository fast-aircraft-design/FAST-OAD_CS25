"""Convenience module for computing leading edge X positions of wing chords."""

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

import numpy as np
import openmdao.api as om


class HTChordGlobalPositions(om.Group):
    """
    Computes leading edge X positions of horizontal tail chords (root, tip)
    with respect to aircraft nose.
    """

    def setup(self):
        self.add_subsystem("compute", ComputeChordGlobalPositions(), promotes=["*"])

        # AddSubtractComp does not allow to set default values of inputs and we do not
        # manage value discrepancies at global level, so we need to do it here, hence
        # the need for the current group.
        self.set_input_defaults(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.set_input_defaults(
            "data:geometry:horizontal_tail:tip:leading_edge:x:local", val=np.nan, units="m"
        )
        self.set_input_defaults(
            "data:geometry:horizontal_tail:MAC:leading_edge:x:local", val=np.nan, units="m"
        )
        self.set_input_defaults("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")


class ComputeChordGlobalPositions(om.Group):
    """
    Computes leading edge X positions of horizontal tail chords with respect to aircraft nose.
    """

    def setup(self):
        self.add_subsystem(
            "MAC25_x",
            om.AddSubtractComp(
                "data:geometry:horizontal_tail:MAC:at25percent:x",
                [
                    "data:geometry:wing:MAC:at25percent:x",
                    "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                ],
                scaling_factors=[1.0, 1.0],
                units="m",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "MAC_x",
            om.AddSubtractComp(
                "data:geometry:horizontal_tail:MAC:leading_edge:x",
                [
                    "data:geometry:horizontal_tail:MAC:at25percent:x",
                    "data:geometry:horizontal_tail:MAC:length",
                ],
                scaling_factors=[1.0, -0.25],
                units="m",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "center_x",
            om.AddSubtractComp(
                "data:geometry:horizontal_tail:center:leading_edge:x",
                [
                    "data:geometry:horizontal_tail:MAC:leading_edge:x",
                    "data:geometry:horizontal_tail:MAC:leading_edge:x:local",
                ],
                scaling_factors=[1.0, -1.0],
                units="m",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "tip_x",
            om.AddSubtractComp(
                "data:geometry:horizontal_tail:tip:leading_edge:x",
                [
                    "data:geometry:horizontal_tail:center:leading_edge:x",
                    "data:geometry:horizontal_tail:tip:leading_edge:x:local",
                ],
                scaling_factors=[1.0, 1.0],
                units="m",
            ),
            promotes=["*"],
        )
