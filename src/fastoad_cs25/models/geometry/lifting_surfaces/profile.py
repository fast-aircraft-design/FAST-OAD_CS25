"""
Management of 2D wing profiles
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

import operator
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..base import Coordinates2D

X = "x"
Z = "z"
THICKNESS = "thickness"


def _rotate_2d_array(
    array: Union[np.ndarray, pd.DataFrame], theta: float
) -> Union[np.ndarray, pd.DataFrame]:
    theta_rad = np.radians(theta)
    rot_matrix = np.array(
        [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]]
    )
    rotated_array = np.matmul(array, rot_matrix)
    if isinstance(rotated_array, pd.DataFrame):
        rotated_array.columns = ["x", "z"]

    return rotated_array


class Profile:
    """Class for managing 2D wing profiles
    :param chord_length:
    :param x:
    :param y:
    """

    # pylint: disable=invalid-name  # X and Z are valid names in this context

    def __init__(self, chord_length: float = 0.0, x: float = 0.0, y: float = 0.0):

        self._rel_mean_line_and_thickness = pd.DataFrame(columns=[X, Z, THICKNESS])
        """
        Relative Mean line and thickness, computed from input data.
        DataFrame keys are 'x', 'z' and 'thickness'
        'x' and 'z' are relative to chord_length
        'thickness' is relative to max thickness
        """

        self.chord_length: float = chord_length
        """ in meters """

        self.max_relative_thickness: float = 0.0
        """ max thickness / chord length"""

        self.twist_angle: float = 0.0
        """ In degrees. Defines how profile is rotated around the 25% chord """

        self.planform_position: Coordinates2D = Coordinates2D(x, y)

    def set_points(
        self,
        x: Sequence,
        z: Sequence,
        keep_chord_length: bool = True,
        keep_relative_thickness: bool = True,
    ):
        """
        Sets points of the 2D profile.

        Provided points are expected to be in order around the profile (clockwise
        or anti-clockwise).

        Provided profile is expected to be at twist angle = 0°.

        :param x: in meters
        :param z: in meters
        :param keep_relative_thickness:
        :param keep_chord_length:
        """

        x = np.asarray(x)
        # Put X of 25% chord position to zero
        x += -0.25 * np.amax(x) - 0.75 * np.amin(x)

        # Separate upper surface from lower surface (easier for computation
        # of thickness and mean line)
        upper, lower = self._create_upper_lower_sides(x, z)

        # Upper and lower sides are defined, we can compute mean line and thickness
        chord_length, max_thickness = self._compute_mean_line_and_thickness(upper, lower)

        if not keep_chord_length or self.chord_length == 0.0:
            self.chord_length = chord_length
        if not keep_relative_thickness or self.max_relative_thickness == 0.0:
            self.max_relative_thickness = max_thickness / chord_length

        # Put Z of mean line at 25% chord to zero
        z_ref = np.interp(
            0.0, self._rel_mean_line_and_thickness[X], self._rel_mean_line_and_thickness[Z]
        )
        self._rel_mean_line_and_thickness[Z] -= z_ref

    def get_mean_line(self) -> pd.DataFrame:
        """Point set of mean line of the profile.

        DataFrame keys are 'x' and 'z', given in meters.
        Returned data are affected by twist angle.
        """
        mean_line = self._rel_mean_line_and_thickness[[X, Z]] * self.chord_length
        return _rotate_2d_array(mean_line, self.twist_angle)

    def get_relative_thickness(self) -> pd.DataFrame:
        """Point set of relative thickness of the profile.

        DataFrame keys are 'x' and 'thickness' and are relative to chord_length.
        'x' is form 0. to 1.
        Warning: in returned DataFrame, data are not affected by twist angle.
        """
        return self._rel_mean_line_and_thickness[[X, THICKNESS]] * [
            1.0,
            self.max_relative_thickness,
        ] + [0.25, 0.0]

    def get_upper_side(self) -> pd.DataFrame:
        """Point set of upper side of the profile.

        DataFrame keys are 'x' and 'z', given in meters.
        Returned data are affected by twist angle.
        """
        return self._get_side_points(operator.add)

    def get_lower_side(self) -> pd.DataFrame:
        """Point set of lower side of the profile.

        DataFrame keys are 'x' and 'z', given in meters.
        Returned data are affected by twist angle.
        """
        return self._get_side_points(operator.sub)

    def _get_side_points(self, operator_) -> pd.DataFrame:
        """
        Computes upper or lower side points.

        operator_ ==  operator.add() -> upper side
        operator_ ==  operator.sub() -> lower side
        """
        mean_line = self._rel_mean_line_and_thickness[[X, Z]]
        half_thickness = pd.DataFrame().reindex_like(mean_line)
        half_thickness[X] = 0.0
        half_thickness[Z] = (
            self._rel_mean_line_and_thickness[THICKNESS] / 2.0 * self.max_relative_thickness
        )
        points = operator_(mean_line, half_thickness) * self.chord_length
        return _rotate_2d_array(points, self.twist_angle)

    def _compute_mean_line_and_thickness(
        self, upper_side_points, lower_side_points
    ) -> Tuple[float, float]:
        """
        Computes mean line and thickness from upper_side_points and lower_side_points.

        Fills self._rel_mean_line_and_thickness with relative values.
        Returns actual chord length and maximum thickness (in meters)
        """
        x = lower_side_points[X].append(upper_side_points[X]).drop_duplicates().sort_values()

        interp_lower = interp1d(lower_side_points[X], lower_side_points[Z], kind="quadratic")
        interp_upper = interp1d(upper_side_points[X], upper_side_points[Z], kind="quadratic")
        z_sides = pd.DataFrame({"z_lower": interp_lower(x), "z_upper": interp_upper(x)})
        z = z_sides.mean(axis=1)
        thickness = z_sides.diff(axis=1).iloc[:, -1]

        chord_length = np.max(x) - np.min(x)
        max_thickness = np.max(thickness)
        self._rel_mean_line_and_thickness = pd.DataFrame(
            {X: x / chord_length, Z: z / chord_length, THICKNESS: thickness / max_thickness}
        )
        return chord_length, max_thickness

    @staticmethod
    def _create_upper_lower_sides(x: Sequence, z: Sequence) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """returns upper side points and lower side points using provided x and z"""
        # FIXME: leading and trailing edges are located roughly. For instance
        #        thick trailing edges are not considered.
        i_leading_edge = np.argmin(x)
        i_trailing_edge = np.argmax(x)

        i1 = min(i_leading_edge, i_trailing_edge)
        i2 = max(i_leading_edge, i_trailing_edge)
        side1 = pd.DataFrame({X: x[i1 : i2 + 1], Z: z[i1 : i2 + 1]})
        side2_1 = pd.DataFrame({X: x[i2:], Z: z[i2:]})
        side2_2 = pd.DataFrame({X: x[: i1 + 1], Z: z[: i1 + 1]})
        side2 = pd.concat((side2_1, side2_2)).reset_index(drop=True)

        side1.sort_values(by=X, inplace=True)
        side2.sort_values(by=X, inplace=True)

        if np.max(side1[Z]) > np.max(side2[Z]):
            upper_side_points = side1
            lower_side_points = side2
        else:
            upper_side_points = side2
            lower_side_points = side1

        upper_side_points.drop_duplicates(inplace=True)
        lower_side_points.drop_duplicates(inplace=True)
        upper_side_points.reset_index(drop=True, inplace=True)
        lower_side_points.reset_index(drop=True, inplace=True)

        return upper_side_points, lower_side_points
