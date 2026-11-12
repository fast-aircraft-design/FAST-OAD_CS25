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

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.assert_utils import assert_check_partials


def assert_exact_derivatives(problem):
    """Checks that the component really uses analytic partials, not finite differences."""
    component = problem.model.component
    component_class = type(component)

    # The component must override compute_partials(), or check_partials()
    # would only see finite-difference derivatives.
    assert component_class.compute_partials is not ExplicitComponent.compute_partials, (
        f"{component_class.__name__} does not implement compute_partials()."
    )

    fd_declarations = [
        (of, wrt)
        for (of, wrt), metadata in component._declared_partials_patterns.items()
        if metadata.get("method") == "fd"
    ]
    # A method="fd" declaration makes check_partials() compare finite
    # differences with themselves, so the test would pass vacuously.
    assert not fd_declarations, (
        f"{component_class.__name__} still declares finite-difference partials: {fd_declarations}"
    )


def assert_component_derivatives(problem, atol=1e-5, rtol=1e-5):
    """Checks analytic partials against finite differences."""
    assert_exact_derivatives(problem)
    data = problem.check_partials(out_stream=None)
    assert_check_partials(data, atol=atol, rtol=rtol)
