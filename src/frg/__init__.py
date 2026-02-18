"""
``frg`` is a Python package to deal with the functional renormalisation group for signal detection. It leverages the action of the RG on different sources of signal to compute a limit of detection and assess the presence of a signal.

The work is based on `theoretical advancements <https://arxiv.org/abs/2201.04250>`__ and previous `numerical works <https://arxiv.org/abs/2310.07499>`__ and `improvements <https://arxiv.org/abs/2507.01064>`__.
"""

# Import the modules
from frg.distributions.distributions import (
    EmpiricalDistribution,
    MarchenkoPastur,
)
from frg.utils.analysis import (
    add_values,
    canonical_dimensions_argsort,
    canonical_dimensions_files,
    canonical_dimensions_ratio_files,
    compute_roi,
    direct_relative_adherence,
    extract_interp_values,
    interp_canonical_dimensions,
    plot_canonical_dimensions,
    plot_canonical_dimensions_scan,
    plot_distribution,
    plot_eigenvalues,
    plot_localization,
    plot_localization_scan,
    plot_potential,
    plot_ratio_scan,
    plot_symmetry_size,
    plot_symmetry_surface,
    plot_trajectories,
)
from frg.utils.utils import get_cfg_defaults, get_logger, load_data

# Set the version number
__version__ = "v2.0.0"

# Set the author
__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"

# Set the license description
__license__ = "CEA Proprietary License"
__url__ = "https://github.com/thesfinox/frg-signal-detection"

# Package imports
__all__ = [
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "__version__",
    "EmpiricalDistribution",
    "MarchenkoPastur",
    "add_values",
    "canonical_dimensions_argsort",
    "canonical_dimensions_files",
    "canonical_dimensions_ratio_files",
    "compute_roi",
    "direct_relative_adherence",
    "extract_interp_values",
    "get_cfg_defaults",
    "get_logger",
    "interp_canonical_dimensions",
    "load_data",
    "plot_canonical_dimensions",
    "plot_canonical_dimensions_scan",
    "plot_distribution",
    "plot_eigenvalues",
    "plot_localization",
    "plot_localization_scan",
    "plot_potential",
    "plot_ratio_scan",
    "plot_symmetry_size",
    "plot_symmetry_surface",
    "plot_trajectories",
]
