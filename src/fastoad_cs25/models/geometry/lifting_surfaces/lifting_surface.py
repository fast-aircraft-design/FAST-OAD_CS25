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
    def __init__(self):
        super().__init__()

        #: List of (coordinate, :class:`Profile`) that compose the lifting surface
        self.sections: List[Profile] = []

        self.mac_position = Coordinates2D(0.0, 0.0)
        self.computed_area: float = 0.0

    def compute_area(self):
        area = pd.DataFrame(
            [
                self._compute_trapezoidal_area(section1, section2)
                for section1, section2 in zip(self.sections[:-1], self.sections[1:])
            ]
        )

        self.computed_area = np.sum(area)

    def compute_mean_aerodynamic_chord(self):
        local_mac_profiles = pd.DataFrame(
            [
                self._compute_trapezoidal_mac(section1, section2)
                for section1, section2 in zip(self.sections[:-1], self.sections[1:])
            ]
        )
        area = pd.DataFrame(
            [
                self._compute_trapezoidal_area(section1, section2)
                for section1, section2 in zip(self.sections[:-1], self.sections[1:])
            ]
        )
        mac = (lambda mac: mac.chord_length)(local_mac_profiles["mac"])
        x = (lambda mac: mac.planform_position.x)(local_mac_profiles["mac"])
        y = (lambda mac: mac.planform_position.y)(local_mac_profiles["mac"])

        self.reference_length = np.dot(mac, area) / np.sum(area)
        self.mac_position.x = np.dot(x, area) / np.sum(area)
        self.mac_position.y = np.dot(y, area) / np.sum(area)

    @staticmethod
    def _compute_trapezoidal_mac(root_profile: Profile, tip_profile: Profile) -> Profile:
        root_chord = root_profile.chord_length
        tip_chord = tip_profile.chord_length
        root_x, root_y = root_profile.planform_position
        tip_x, tip_y = tip_profile.planform_position

        taper_ratio = root_chord / tip_chord

        mac = Profile()
        mac.chord_length = (
            2.0 / 3.0 * root_chord * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
        )

        coeff = (1.0 + 2.0 * taper_ratio) / (3.0 + 3.0 * taper_ratio)
        mac.planform_position.x = root_x + (tip_x - root_x) * coeff
        mac.planform_position.x = root_y + (tip_y - root_y) * coeff

        return mac

    @staticmethod
    def _compute_trapezoidal_area(root_profile: Profile, tip_profile: Profile) -> float:
        root_chord = root_profile.chord_length
        tip_chord = tip_profile.chord_length
        root_y = root_profile.planform_position.y
        tip_y = tip_profile.planform_position.y

        return (tip_y - root_y) * (root_chord + tip_chord) / 2.0
