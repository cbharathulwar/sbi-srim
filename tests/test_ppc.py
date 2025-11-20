import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
from src.utils.srim_parser import summarize_all_runs
from src.utils.analysis_utils import plot_ppc_histograms_per_track
from pathlib import Path


def test_summarize_and_ppc_does_not_crash(tmp_path):
    dummy = pd.DataFrame(
        {
            "track_id": ["t1"],
            "theta_eV": [1000.0],
            "mean_depth_A": [1.0],
            "std_depth_A": [0.1],
            "vacancies_per_ion": [10.0],
        }
    )
    dummy.to_csv(tmp_path / "dummy.csv", index=False)

    plot_ppc_histograms_per_track(
        df=dummy,
        observed={
            "t1": {"mean_depth_A": 1.0, "std_depth_A": 0.1, "vacancies_per_ion": 10.0}
        },
        output_dir=str(tmp_path),
    )
