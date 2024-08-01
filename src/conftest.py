"""
Basic settings for tests
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

# Note: this file has to be put in src/, not in project root folder, to ensure that
# `pytest src` will run OK after a `pip install .`

import os.path as pth
from platform import system
from shutil import which
from typing import Optional

import pytest


@pytest.fixture(autouse=True)
def no_xfoil_skip(request, xfoil_path):
    """
    Use @pytest.mark.skip_if_no_xfoil() before a test to skip it if xfoil_path
    fixture returns None and OS is not Windows.
    """
    if request.node.get_closest_marker("skip_if_no_xfoil"):
        if xfoil_path is None and system() != "Windows":
            pytest.skip("No XFOIL executable available")


@pytest.fixture
def xfoil_path() -> Optional[str]:
    """
    On a system that is not Windows, a XFOIL executable with name "xfoil" can
    be put in "<project_root>/tests/xfoil_exe/".
    In this case, this fixture will return its path.

    :return: The path of the XFOIL executable
    """
    if system() == "Windows":
        # On Windows, we use the embedded executable
        return None

    path = pth.abspath(pth.join("tests/xfoil_exe", "xfoil"))
    if pth.exists(path):
        # If there is a local xfoil, use it
        return path

    # Otherwise, use one that is in PATH, if it exists
    return which("xfoil")
