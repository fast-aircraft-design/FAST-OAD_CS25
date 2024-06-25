"""
Test module for XFOIL component
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2024 ONERA & ISAE-SUPAERO
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

# pylint: disable=redefined-outer-name  # needed for fixtures

import os.path as pth
import shutil

import pytest
from fastoad.testing import run_system
from openmdao.core.indepvarcomp import IndepVarComp

from ..xfoil_polar import DEFAULT_2D_CL_MAX, XfoilPolar

XFOIL_RESULTS = pth.join(pth.dirname(__file__), "results")


@pytest.mark.skip_if_no_xfoil()
def test_compute(xfoil_path):
    """Tests a simple XFOIL run"""

    if pth.exists(XFOIL_RESULTS):
        shutil.rmtree(XFOIL_RESULTS)

    ivc = IndepVarComp()
    ivc.add_output("xfoil:reynolds", 18000000)
    ivc.add_output("xfoil:mach", 0.20)
    ivc.add_output("data:geometry:wing:thickness_ratio", 0.1284)

    # Base test ----------------------------------------------------------------
    xfoil_comp = XfoilPolar(
        alpha_start=15.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(1.94, 1e-2)
    assert not pth.exists(XFOIL_RESULTS)

    # Test that will stop before real max CL -----------------------------------
    xfoil_comp = XfoilPolar(
        alpha_start=12.0, alpha_end=20.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(1.92, 1e-2)
    assert not pth.exists(XFOIL_RESULTS)

    # Test that will not converge ----------------------------------------------
    xfoil_comp = XfoilPolar(
        alpha_start=50.0, alpha_end=55.0, iter_limit=2, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(DEFAULT_2D_CL_MAX, 1e-2)
    assert not pth.exists(XFOIL_RESULTS)

    # Test that will output results in provided folder -------------------------
    xfoil_comp = XfoilPolar(
        iter_limit=20, result_folder_path=XFOIL_RESULTS, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(1.94, 1e-2)
    assert pth.exists(XFOIL_RESULTS)
    assert pth.exists(pth.join(XFOIL_RESULTS, "polar_result.txt"))


@pytest.mark.skip_if_no_xfoil()
def test_compute_with_provided_path(xfoil_path):
    """Test that option "use_exe_path" works"""
    ivc = IndepVarComp()
    ivc.add_output("xfoil:reynolds", 18000000)
    ivc.add_output("xfoil:mach", 0.20)
    ivc.add_output("data:geometry:wing:thickness_ratio", 0.1284)

    xfoil_comp = XfoilPolar(alpha_start=18.0, alpha_end=21.0, iter_limit=20)
    xfoil_comp.options["xfoil_exe_path"] = "Dummy"  # bad name
    with pytest.raises(ValueError):
        problem = run_system(xfoil_comp, ivc)

    xfoil_comp.options["xfoil_exe_path"] = (
        xfoil_path
        if xfoil_path
        else pth.join(pth.dirname(__file__), pth.pardir, "xfoil699", "xfoil.exe")
    )
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(1.94, 1e-2)
