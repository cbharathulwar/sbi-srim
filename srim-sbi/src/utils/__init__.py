"""
utils
=====

Utility package for the SRIM–SBI pipeline.

Submodules
----------
- data_utils       : data prep, random track selection, observed vectors
- srim_utils       : running SRIM (single + batch)
- sbi_runner       : SBI training/sampling helpers
- srim_parser      : parsing SRIM outputs into summaries
- analysis_utils   : PPC plotting / analysis utilities

The package also re-exports common top-level helpers for convenience, e.g.:

    from utils import pick_random_tracks, run_srim_batch, sample_posterior_bulk
"""

# Always expose submodules
from . import data_utils as data_utils
from . import srim_utils as srim_utils
from . import sbi_runner as sbi_runner
from . import srim_parser as srim_parser
from . import analysis_utils as analysis_utils


# Build __all__ dynamically
__all__ = [
    "data_utils",
    "srim_utils",
    "sbi_runner",
    "srim_parser",
    "analysis_utils",
]

_for_export = [
    "pick_tracks_deterministic",
    "create_observed_dataframe",
    "run_srim_for_theta",
    "run_srim_batch",
    "sample_posterior_bulk",
    "make_prior",
    "make_inference",
    "train_posterior",
    "summarize_srim_output",
    "summarize_all_runs",
    "plot_ppc_histograms",
    "run_srim_multi_track",
    "sample_posterior_theta",
    'tensor_to_observed_dict',
    'clean_summary_data'
]

__all__ += [name for name in _for_export if name in globals()]