import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.utils.data_utils import preprocess, make_x_test
from pathlib import Path

RAW_CSV = Path("data/all_vacancies.csv")


def test_preprocess_loads_data():
    x_obs, theta, track_ids, grouped, df_summary = preprocess(RAW_CSV)
    assert len(x_obs) > 0
    assert "energy_keV" in df_summary.columns
    assert x_obs.shape[0] == theta.shape[0]


def test_make_x_test_balanced():
    _, _, _, _, df_summary = preprocess(RAW_CSV)
    x_test, ids = make_x_test(df_summary, n_per_energy=1)
    assert len(x_test) > 0
    assert "energy_keV" in x_test.columns
