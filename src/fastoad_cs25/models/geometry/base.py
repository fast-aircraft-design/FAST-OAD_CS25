"""
Module for basic geometrical data
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

from abc import ABC
from collections import namedtuple
from typing import Optional, TypeVar

import numpy as np

ComponentGeometrySubClass = TypeVar("ComponentGeometrySubClass", bound="ComponentGeometry")

Coordinates2D = namedtuple("Coordinates2D", ["x", "y"])
Coordinates3D = namedtuple("Coordinates3D", ["x", "y", "z"])


class ComponentGeometry(ABC):
    """A class for storing geometry data of an aircraft component"""

    def __init__(self):
        self.reference_length: float = np.nan
        """ Reference length of the element """

        self.reference_area: float = np.nan
        """ Reference area of the element """

        self.length: float = np.nan
        """ X-wise size of the element """

        self.width: float = np.nan
        """ Y-size size of the element """

        self.height: Optional[float] = np.nan
        """ Z-wise size of the element """
