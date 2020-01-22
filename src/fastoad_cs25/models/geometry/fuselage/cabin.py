""" Module for geometry of the cabin """
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

from collections import namedtuple

import numpy as np

Seat = namedtuple("Seat", ["width", "length"])
""" The length of the seat includes the space to the one before"""


class Cabin:
    """
    Class for assessing geometry of aircraft cabin
    """

    def __init__(self):
        super().__init__()

        self.eco_seat: Seat = Seat(0.457, 0.864)
        """ Dimensions of an economical class passenger seat (in meters) """

        self.eco_seat_count_in_row: int = np.nan
        """ Number of economical class passenger seats in a row """

        self.aisle_width: float = np.nan
        """ width of aisle (in meters) """

        self.aisle_count: int = np.nan
        """ count of aisles """

        self.emergency_exit_width: float = np.nan
        """ width of an emergency exit (in meters) """

        self.emergency_exit_count: int = np.nan
        """ count of emergency exits between seat rows on one side"""

        self.npax_eco_only: int = np.nan
        """ passenger count, assuming only economical class seats """

    @property
    def width(self):
        """width of cabin (computed) in meters"""
        armchair_width = 0.051  # 2 inches
        return (
            self.eco_seat_count_in_row * self.eco_seat.width
            + self.aisle_count * self.aisle_width
            + armchair_width * (self.eco_seat_count_in_row + self.aisle_count + 1)
            + armchair_width
        )  # 1 inch margin on each side

    @property
    def length(self):
        """length of cabin (computed) in meters"""
        return (
            self.get_row_count() * self.eco_seat.length
            + self.emergency_exit_count * self.emergency_exit_width
        )

    def get_row_count(self):
        """
        :return: count of seat rows
        """
        return int(self.npax_eco_only / self.eco_seat_count_in_row)

    def get_commercial_crew_count(self):
        """
        :return: estimate of needed count of commercial crew members
        """
        return int((self.npax_eco_only + 17) / 35)

    def _compute_row_count(self):
        initial_row_count = int(self.npax_eco_only / self.eco_seat_count_in_row)
        out_of_cyl_row_count = 2 * self.eco_seat_count_in_row - 4
        reduced_row_count = 2 * self.eco_seat_count_in_row - 11
        left_seat_count = reduced_row_count + self.npax_eco_only % self.eco_seat_count_in_row
        total_row_count = initial_row_count + int(left_seat_count) / self.eco_seat_count_in_row