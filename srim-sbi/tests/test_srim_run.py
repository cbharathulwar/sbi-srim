import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, pandas as pd
from src.utils.srim_utils import run_srim_multi_track
from pathlib import Path


def test_run_srim_multi_track_creates_output(tmp_path):
    x_test = pd.DataFrame(
        [
            {
                "track_id": "t1",
                "mean_depth_A": 1,
                "std_depth_A": 1,
                "vacancies_per_ion": 1,
                "energy_keV": 1000,
                "ion": "C",
            }
        ]
    )
    samples_dict = {"t1": torch.tensor([1000.0])}

    # use your real SRIM path here
    srim_dir = Path("/Users/yourname/Documents/Research/Walsworth/SRIM-2013")

    run_srim_multi_track(
        samples_dict=samples_dict,
        x_test=x_test,
        track_ids=["t1"],
        srim_directory=str(srim_dir),
        output_base=tmp_path,
        ion_symbol="C",
        number_ions=1,
        df_summary=x_test,
        overwrite=True,
    )

    out = list(tmp_path.glob("track_*/*metadata.json"))
    assert len(out) == 1, "No SRIM metadata.json found!"
