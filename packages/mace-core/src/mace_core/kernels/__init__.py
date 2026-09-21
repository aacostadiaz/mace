"""The kernel backend contract: descriptors, capabilities, protocol, registry.

Imports no framework. Everything here is the statement of what an operation is
and what a backend can do, expressed over strings and numbers, so that the same
contract serves the torch and the jax implementations and can be written into a
checkpoint.

Three rules this package exists to enforce, each replacing something the frozen
tree does the other way round:

* **A backend declares, it never chooses.** On the frozen tree a host without
  ``cuequivariance`` silently switches the Clebsch-Gordan basis and trains a
  different network. Here the basis is a recorded field and a backend that
  cannot serve it fails the build loudly.
* **Dispatch resolves once, at build time.** The frozen tree branches inside
  the interaction blocks on whether an attribute is set, and monkeypatches
  ``forward`` in three places. Here a descriptor goes in, an op comes out, and
  the op is frozen into the module tree.
* **One canonical weight layout.** The frozen tree has five conversion command
  line tools because each backend stores weights its own way.
"""

from mace_core.kernels.canonical import (
    CANONICAL_LAYOUT,
    KERNEL_SPEC_VERSION,
    canonical_weight_shape,
    contraction_path_labels,
    contraction_path_order,
)
from mace_core.kernels.capabilities import (
    BackendCapabilities,
    UnsupportedDescriptorError,
)
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.precision import PRECISIONS, Precision
from mace_core.kernels.protocol import (
    DISPATCHED_OPS,
    REFERENCE_ONLY_OPS,
    KernelBackend,
)
from mace_core.kernels.registry import (
    ENTRY_POINT_GROUPS,
    BackendNotAvailableError,
    DiscoveredBackend,
    available_backends,
    get_backend,
)
from mace_core.kernels.reorder import (
    ReorderBlock,
    ReorderError,
    WeightReorder,
    derive_reorder,
)

__all__ = [
    "CANONICAL_LAYOUT",
    "DISPATCHED_OPS",
    "ENTRY_POINT_GROUPS",
    "KERNEL_SPEC_VERSION",
    "PRECISIONS",
    "REFERENCE_ONLY_OPS",
    "BackendCapabilities",
    "BackendNotAvailableError",
    "ChannelwiseTPConvDescriptor",
    "Descriptor",
    "DiscoveredBackend",
    "FullyConnectedTPDescriptor",
    "KernelBackend",
    "LinearDescriptor",
    "Precision",
    "RadialBasisDescriptor",
    "ReorderBlock",
    "ReorderError",
    "SegmentReduceDescriptor",
    "SphericalHarmonicsDescriptor",
    "SymmetricContractionDescriptor",
    "UnsupportedDescriptorError",
    "WeightReorder",
    "available_backends",
    "canonical_weight_shape",
    "contraction_path_labels",
    "contraction_path_order",
    "derive_reorder",
    "get_backend",
]
