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

from fastoad.modules.geometry import ComponentGeometry

Seat = namedtuple("Seat", ["width", "length"])


class Cabin(ComponentGeometry):
    def __init__(self):
        self.eco_seat: Seat = Seat(32.0, 20.0)
        """ Dimensions of an economy class passenger seat (in meters) """

        self.eco_seat_count_in_row: int = 0
        """ Number of economy class passenger seats in a row """

        self.aisle_width: float = 0.0
        """ width of aisle (in meters) """

        self.aisle_count: int = 0
        """ count of aisles """

        self.emergency_exit_width: float = 0.0
        """ width of an emergency exit (in meters) """

        self.emergency_exit_count: int = 0
        """ count of emergency exits """

        self.npax: int = 0
        """ passenger count, assuming only economy class seats"""

    def get_width(self):
        armchair_width = 0.051  # 2 inches
        return (
            self.eco_seat_count_in_row * self.eco_seat.width
            + self.aisle_count * self.aisle_width
            + armchair_width * (self.eco_seat_count_in_row + self.aisle_count + 1)
            + armchair_width
        )  # some margin ?

    def get_row_count(self):
        return int(self.npax / self.eco_seat_count_in_row)

    def get_length(self):
        return (
            self.get_row_count() * self.eco_seat.length
            + self.emergency_exit_count * self.emergency_exit_width
        )


class Fuselage(ComponentGeometry):
    """Class for fuselage geometry of standard commercial transport aircraft"""

    pass
