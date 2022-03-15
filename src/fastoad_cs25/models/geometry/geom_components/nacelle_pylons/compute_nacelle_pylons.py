"""
    Estimation of nacelle and pylon geometry
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

from dataclasses import dataclass

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from ...constants import SERVICE_NACELLE_PYLON_GEOMETRY


@dataclass
class Chord:
    """Container for storing chord length and x,y coordinates of leading edge."""

    x: float
    y: float
    length: float


@dataclass
class Nacelle:
    """Simple class for computing nacelle geometry"""

    max_thrust: float

    @property
    def diameter(self):
        """Nacelle diameter in m."""
        # FIXME: use output of engine module
        return 0.00904 * np.sqrt(self.max_thrust * 0.225) + 0.7

    @property
    def length(self):
        """Nacelle length in m."""
        # FIXME: use output of engine module
        return 0.032 * np.sqrt(self.max_thrust * 0.225)

    @property
    def wetted_area(self):
        """Wetted area for one nacelle in m**2."""
        return 0.0004 * self.max_thrust * 0.225 + 11


@oad.RegisterSubmodel(
    SERVICE_NACELLE_PYLON_GEOMETRY, "fastoad.submodel.geometry.nacelle_and_pylon.legacy"
)
class ComputeNacelleAndPylonsGeometry(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Nacelle and pylon geometry estimation"""

    def setup(self):

        self.add_input("data:propulsion:MTO_thrust", val=np.nan, units="N")
        self.add_input("data:geometry:propulsion:engine:y_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:propulsion:pylon:length", units="m")
        self.add_output("data:geometry:propulsion:fan:length", units="m")
        self.add_output("data:geometry:propulsion:nacelle:length", units="m")
        self.add_output("data:geometry:propulsion:nacelle:diameter", units="m")
        self.add_output("data:geometry:landing_gear:height", units="m")
        self.add_output("data:geometry:propulsion:nacelle:y", units="m")
        self.add_output("data:geometry:propulsion:pylon:wetted_area", units="m**2")
        self.add_output("data:geometry:propulsion:nacelle:wetted_area", units="m**2")
        self.add_output("data:weight:propulsion:engine:CG:x", units="m")

    def setup_partials(self):
        self.declare_partials(
            "data:geometry:propulsion:nacelle:diameter", "data:propulsion:MTO_thrust", method="fd"
        )
        self.declare_partials(
            "data:geometry:propulsion:nacelle:length", "data:propulsion:MTO_thrust", method="fd"
        )
        self.declare_partials(
            "data:geometry:landing_gear:height", "data:propulsion:MTO_thrust", method="fd"
        )
        self.declare_partials(
            "data:geometry:propulsion:fan:length", "data:propulsion:MTO_thrust", method="fd"
        )
        self.declare_partials(
            "data:geometry:propulsion:pylon:length", "data:propulsion:MTO_thrust", method="fd"
        )
        self.declare_partials(
            "data:geometry:propulsion:nacelle:y",
            [
                "data:propulsion:MTO_thrust",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:propulsion:engine:y_ratio",
                "data:geometry:wing:span",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:weight:propulsion:engine:CG:x",
            [
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:leading_edge:x:local",
                "data:geometry:wing:kink:leading_edge:x:local",
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:root:y",
                "data:geometry:wing:kink:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:kink:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:fuselage:length",
                "data:propulsion:MTO_thrust",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:propulsion:engine:y_ratio",
                "data:geometry:wing:span",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:propulsion:nacelle:wetted_area",
            "data:propulsion:MTO_thrust",
            method="fd",
        )
        self.declare_partials(
            "data:geometry:propulsion:pylon:wetted_area", "data:propulsion:MTO_thrust", method="fd"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_layout = np.round(inputs["data:geometry:propulsion:layout"])
        root_chord = Chord(
            x=None,
            y=inputs["data:geometry:wing:root:y"],
            length=inputs["data:geometry:wing:root:chord"],
        )
        kink_chord = Chord(
            x=inputs["data:geometry:wing:kink:leading_edge:x:local"],
            y=inputs["data:geometry:wing:kink:y"],
            length=inputs["data:geometry:wing:kink:chord"],
        )
        tip_chord = Chord(
            x=inputs["data:geometry:wing:tip:leading_edge:x:local"],
            y=inputs["data:geometry:wing:tip:y"],
            length=inputs["data:geometry:wing:tip:chord"],
        )

        nacelle = Nacelle(inputs["data:propulsion:MTO_thrust"])
        outputs["data:geometry:propulsion:nacelle:length"] = nacelle.length
        outputs["data:geometry:propulsion:nacelle:diameter"] = nacelle.diameter
        outputs["data:geometry:propulsion:nacelle:wetted_area"] = nacelle.wetted_area

        outputs["data:geometry:propulsion:pylon:length"] = 1.1 * nacelle.length
        outputs["data:geometry:propulsion:fan:length"] = 0.60 * nacelle.length

        y_nacelle = self._compute_nacelle_y(nacelle, inputs, propulsion_layout)

        if propulsion_layout == 1:
            if y_nacelle <= kink_chord.y:
                x_nacelle_cg = self._get_nacelle_cg_x(nacelle, y_nacelle, root_chord, kink_chord)
            else:
                x_nacelle_cg = self._get_nacelle_cg_x(nacelle, y_nacelle, kink_chord, tip_chord)
            x_nacelle_cg_absolute = (
                inputs["data:geometry:wing:MAC:at25percent:x"]
                - 0.25 * inputs["data:geometry:wing:MAC:length"]
                - (inputs["data:geometry:wing:MAC:leading_edge:x:local"] - x_nacelle_cg)
            )
        elif propulsion_layout == 2:
            x_nacelle_cg_absolute = 0.8 * inputs["data:geometry:fuselage:length"]
        else:
            raise ValueError("Value of data:geometry:propulsion:layout can only be 1 or 2")

        outputs["data:geometry:propulsion:nacelle:y"] = y_nacelle
        outputs["data:weight:propulsion:engine:CG:x"] = x_nacelle_cg_absolute

        outputs["data:geometry:propulsion:pylon:wetted_area"] = 0.35 * nacelle.wetted_area
        outputs["data:geometry:landing_gear:height"] = 1.4 * (
            0.00904 * np.sqrt(inputs["data:propulsion:MTO_thrust"] * 0.225) + 0.7
        )

    @staticmethod
    def _get_nacelle_cg_x(nacelle: Nacelle, y_nacelle: float, chord1: Chord, chord2: Chord):
        chord_at_engine_location = chord2.length + (chord1.length - chord2.length) * (
            chord2.y - y_nacelle
        ) / (chord2.y - chord1.y)
        delta_x_nacelle = 0.05 * chord_at_engine_location
        x_nacelle_cg = (
            chord2.x * (y_nacelle - chord1.y) / (chord2.y - chord1.y)
            - delta_x_nacelle
            - 0.2 * nacelle.length
        )
        return x_nacelle_cg

    @staticmethod
    def _compute_nacelle_y(nacelle, inputs, propulsion_layout):
        if propulsion_layout == 1:
            y_nacelle = (
                inputs["data:geometry:propulsion:engine:y_ratio"]
                * inputs["data:geometry:wing:span"]
                / 2.0
            )
        elif propulsion_layout == 2:
            y_nacelle = (
                inputs["data:geometry:fuselage:maximum_width"] / 2.0 + 0.5 * nacelle.diameter + 0.7
            )
        else:
            raise ValueError("Value of data:geometry:propulsion:layout can only be 1 or 2")
        return y_nacelle
