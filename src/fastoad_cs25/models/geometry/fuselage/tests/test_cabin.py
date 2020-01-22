""" Test module for cabin.py """
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

import pytest
from numpy.testing import assert_allclose

from ..cabin import Cabin, Seat


# pylint: disable=redefined-outer-name  # false positive on pytest fixtures


@pytest.fixture
def sample_cabin() -> Cabin:
    cabin = Cabin()
    cabin.npax_eco_only = 157
    cabin.eco_seat_count_in_row = 6
    cabin.eco_seat = Seat(0.457, 0.864)
    cabin.aisle_width = 0.48
    cabin.aisle_count = 1
    cabin.emergency_exit_width = 0.51
    cabin.emergency_exit_count = 1
    return cabin


def test_width(sample_cabin):
    """test of Cabin.width property"""
    assert_allclose(sample_cabin.width, 3.68, atol=1e-2)


def test_length(sample_cabin):
    """test of Cabin.length property"""
    assert_allclose(sample_cabin.length, 22.87, atol=1e-2)
