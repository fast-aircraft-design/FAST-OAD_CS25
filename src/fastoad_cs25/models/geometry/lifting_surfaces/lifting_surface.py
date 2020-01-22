"""
Geometry of generic lifting surfaces
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

from typing import List

import numpy as np
import pandas as pd

from .profile import Profile
from .. import ComponentGeometry
from ..base import Coordinates2D


class LiftingSurface(ComponentGeometry):
    """Class for geometry of generic lifting surface"""

    def __init__(self):
        super().__init__()

        #: List of (coordinate, :class:`Profile`) that compose the lifting surface
        self.sections: List[Profile] = []
        """ List of :class:`Profile` that compose the lifting surface """

        self.mac_position = Coordinates2D(0.0, 0.0)
        self.planform_area: float = 0.0

    def compute_planform_area(self) -> float:
        """
        Computes planform area as the sum of trapezoid areas between sections

        Computed area is stored in self.planform_area then returned.

        :return: planform area in square meters
        """
        area = pd.DataFrame(
            [
                self._compute_trapezoid_area(section1, section2)
                for section1, section2 in zip(self.sections[:-1], self.sections[1:])
            ]
        )

        self.planform_area = np.sum(area).item()
        return self.planform_area

    def compute_mean_aerodynamic_chord(self):
        """
        Computes Mean Aerodynamic Chord area as the sum of trapezoid areas between sections

        Computed MAC is stored in self.reference_length then returned.
        MAC position is also stored in self.mac_position.

        :return: MAC in meters
        """
        local_mac_profiles = pd.Series(
            [
                self._compute_trapezoid_mac(section1, section2)
                for section1, section2 in zip(self.sections[:-1], self.sections[1:])
            ]
        )
        area = pd.Series(
            [
                self._compute_trapezoid_area(section1, section2)
                for section1, section2 in zip(self.sections[:-1], self.sections[1:])
            ]
        )
        local_mac = local_mac_profiles.apply(lambda p: p.chord_length)
        local_mac_x = local_mac_profiles.apply(lambda p: p.planform_position.x)
        local_mac_y = local_mac_profiles.apply(lambda p: p.planform_position.y)

        self.reference_length = np.dot(local_mac, area) / np.sum(area)
        self.mac_position = Coordinates2D(
            np.dot(local_mac_x, area) / np.sum(area), np.dot(local_mac_y, area) / np.sum(area)
        )

    @staticmethod
    def _compute_trapezoid_mac(root_profile: Profile, tip_profile: Profile) -> Profile:
        """
        Computes the Mean Aerodynamic Chord of the trapezoid part between 2 sections.

        :return: a :class:`Profile` instance that contains length and position of the MAC
        """
        root_chord = root_profile.chord_length
        tip_chord = tip_profile.chord_length
        root_x, root_y = root_profile.planform_position
        tip_x, tip_y = tip_profile.planform_position

        taper_ratio = tip_chord / root_chord

        mac = Profile()
        mac.chord_length = (
            2.0 / 3.0 * root_chord * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
        )

        coeff = (1.0 + 2.0 * taper_ratio) / (3.0 + 3.0 * taper_ratio)
        mac.planform_position = Coordinates2D(
            root_x + (tip_x - root_x) * coeff, root_y + (tip_y - root_y) * coeff
        )

        return mac

    @staticmethod
    def _compute_trapezoid_area(root_profile: Profile, tip_profile: Profile) -> float:
        """
        Computes the area of the trapezoid part between 2 sections.

        :return: the area in squared meters
        """
        root_chord = root_profile.chord_length
        tip_chord = tip_profile.chord_length
        root_y = root_profile.planform_position.y
        tip_y = tip_profile.planform_position.y

        return (tip_y - root_y) * (root_chord + tip_chord) / 2.0
