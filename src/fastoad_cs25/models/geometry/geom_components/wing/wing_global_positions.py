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

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from .constants import SERVICE_WING_GEOMETRY_GLOBAL_POSITIONS


@oad.RegisterSubmodel(
    SERVICE_WING_GEOMETRY_GLOBAL_POSITIONS,
    "fastoad.submodel.geometry.wing.global_chord_positions.legacy",
)
class ChordGlobalPositions(om.Group):
    """
    Computes leading edge X positions of wing chords (oot, kink, tip) with respect to aircraft nose.
    """

    def setup(self):
        self.add_subsystem("compute", ComputeChordGlobalPositions(), promotes=["*"])

        # AddSubtractComp does not allow to set default values of inputs and we do not
        # manage value discrepancies at global level, so we need to do it here, hence
        # the need for the current group.
        for chord_id in ["kink", "tip"]:
            self.set_input_defaults(
                f"data:geometry:wing:{chord_id}:leading_edge:x:local", val=np.nan, units="m"
            )
        self.set_input_defaults("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.set_input_defaults(
            "data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m"
        )
        self.set_input_defaults("data:geometry:wing:MAC:length", val=np.nan, units="m")


class ComputeChordGlobalPositions(om.AddSubtractComp):
    """
    Computes leading edge X positions of wing chords with respect to aircraft nose.
    """

    def setup(self):
        self.add_equation(
            "data:geometry:wing:root:leading_edge:x",
            [
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:wing:MAC:leading_edge:x:local",
                "data:geometry:wing:MAC:length",
            ],
            scaling_factors=[1.0, -1.0, -0.25],
            units="m",
        )
        self.add_equation(
            "data:geometry:wing:MAC:leading_edge:x",
            [
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:wing:MAC:length",
            ],
            scaling_factors=[1.0, -0.25],
            units="m",
        )
        for chord_id in ["kink", "tip"]:
            self.add_equation(
                f"data:geometry:wing:{chord_id}:leading_edge:x",
                [
                    f"data:geometry:wing:{chord_id}:leading_edge:x:local",
                    "data:geometry:wing:MAC:at25percent:x",
                    "data:geometry:wing:MAC:leading_edge:x:local",
                    "data:geometry:wing:MAC:length",
                ],
                scaling_factors=[1.0, 1.0, -1.0, -0.25],
                units="m",
            )
