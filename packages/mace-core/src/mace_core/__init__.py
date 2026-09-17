"""Framework-agnostic contract and pure math for MACE v1.

This package imports no framework: everything here is plain Python, numpy, ase
and pydantic, so the same types serve the torch and the jax implementations
alike.

Most of the public surface lives in the submodules and is imported from them
rather than from here, because some of them are not cheap to import and a
caller that only wants a unit constant should not pay for a file parser:

``mace_core.data``
    :class:`~mace_core.data.configuration.Configuration`, the boundary object
    of the data layer, its key specification, and the parsing and splitting
    functions over it.

``mace_core.elements``
    the default property keys and the element index table.

``mace_core.units``
    unit constants, and the single statement of each physics sign convention.

``mace_core.clebsch_gordan``
    the reduced symmetric tensor-product basis, its pinned path order and
    per-path normalization, and the full/reduced conversions.

The typed output object and the observable declarations are re-exported below,
because they are the contract every model and every consumer is written
against and they cost only pydantic to import.
"""

from importlib.metadata import PackageNotFoundError, version

from mace_core.observables import (
    DerivativeSpec,
    InputSpec,
    ObservableCatalogue,
    ObservableSpec,
    load_default_catalogue,
)
from mace_core.outputs import MACEOutput

__all__ = [
    "DerivativeSpec",
    "InputSpec",
    "MACEOutput",
    "ObservableCatalogue",
    "ObservableSpec",
    "__version__",
    "load_default_catalogue",
]

#: Version of the installed `mace-core` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved.
try:
    __version__ = version("mace-core")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
