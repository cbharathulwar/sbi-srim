import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
from src.utils.data_utils import preprocess
from src.utils.sbi_runner import (
    make_prior,
    make_inference,
    train_posterior,
    sample_posterior_bulk,
)


def test_posterior_training_and_sampling():
    x_obs, theta, *_ = preprocess("data/all_vacancies.csv")
    prior = make_prior([1_000.0], [2_000_000.0])
    inference = make_inference(prior, density_estimator="nsf")
    posterior = train_posterior(inference, theta[:20], x_obs[:20])
    x_test = torch.tensor(x_obs[:2], dtype=torch.float32)
    samples_dict, _ = sample_posterior_bulk(
        posterior, x_test, 5, track_ids=["t1", "t2"]
    )
    assert all(k in samples_dict for k in ["t1", "t2"])
