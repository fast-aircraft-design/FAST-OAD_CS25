""" Test module for lifting_surface.py """
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

from ..lifting_surface import LiftingSurface
from ..profile import Profile


# pylint: disable=redefined-outer-name  # false positive on pytest fixtures


@pytest.fixture
def trapezoid():
    """Defines a wing with a trapezoidal planform"""
    surf = LiftingSurface()
    surf.sections.append(Profile(3.0, 0.0, 0.0))
    surf.sections.append(Profile(1.5, 10.0, 0.5))
    return surf


@pytest.fixture
def wing():
    """Defines a "classical" wing with a kink"""
    surf = LiftingSurface()
    surf.sections.append(Profile(6.260, 0.0, 0.000))  # center fuselage
    surf.sections.append(Profile(6.260, 0.0, 1.960))  # l2
    surf.sections.append(Profile(3.985, 2.275, 6.321))  # l3
    surf.sections.append(Profile(1.882, 7.222, 15.801))  # l4
    return surf


def test_compute_planform_area(trapezoid, wing):
    """test of LiftingSurface.compute_planform_area()"""
    trapezoid.compute_planform_area()
    assert_allclose(1.125, trapezoid.planform_area)

    wing.compute_planform_area()
    assert_allclose(62.418, wing.planform_area, atol=1e-3)


def test_compute_mean_aerodynamic_chord(trapezoid, wing):
    """test of LiftingSurface.compute_mean_aerodynamic_chord()"""
    trapezoid.compute_mean_aerodynamic_chord()
    assert_allclose(2.0 / 3.0 * 3.0 * (1 + 0.5 + 0.5**2) / 1.5, trapezoid.reference_length)

    wing.compute_mean_aerodynamic_chord()
    assert_allclose(4.457, wing.reference_length, atol=1e-3)
    assert_allclose((2.361, 6.293), wing.mac_position, atol=1e-3)
