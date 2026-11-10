"""Utilities for analytic derivative checks."""

#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2026 ONERA & ISAE-SUPAERO
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

import warnings
from pathlib import Path

import numpy as np
from fastoad.io import VariableIO
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.assert_utils import assert_check_partials


def get_ivc_from_xml(xml_path: Path):
    """Reads all variables of an XML file and returns an IndepVarComp."""
    reader = VariableIO(str(xml_path))
    reader.path_separator = ":"
    return reader.read().to_ivc()


def assert_model_is_error_free(problem):
    """Runs the model and checks that outputs are finite and no RuntimeWarning is raised."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        problem.run_model()

    runtime_warnings = [
        warning for warning in caught_warnings if issubclass(warning.category, RuntimeWarning)
    ]
    assert not runtime_warnings, (
        "The model raised RuntimeWarning(s) while being evaluated: "
        f"{[str(warning.message) for warning in runtime_warnings]}"
    )

    component = problem.model.component
    for output_name in component._var_rel_names["output"]:
        value = problem.get_val(output_name)
        assert np.all(np.isfinite(value)), f"Output '{output_name}' is not finite: {value}"


def assert_exact_derivatives(problem):
    """Checks that the component really uses analytic partials, not finite differences."""
    component = problem.model.component
    component_class = type(component)

    assert component_class.compute_partials is not ExplicitComponent.compute_partials, (
        f"{component_class.__name__} does not implement compute_partials()."
    )

    fd_declarations = [
        (of, wrt)
        for (of, wrt), metadata in component._declared_partials_patterns.items()
        if metadata.get("method") == "fd"
    ]
    assert not fd_declarations, (
        f"{component_class.__name__} still declares finite-difference partials: {fd_declarations}"
    )


def assert_component_derivatives(problem, atol=1e-5, rtol=1e-6):
    """Checks model evaluation and analytic-vs-finite-difference partials."""
    assert_model_is_error_free(problem)
    assert_exact_derivatives(problem)
    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=atol, rtol=rtol)
