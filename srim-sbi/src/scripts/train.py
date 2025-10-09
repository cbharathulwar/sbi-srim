#!/usr/bin/env python3
"""
Train SRIM–SBI Pipeline Runner
=============================

End-to-end runner for:
    1. Data preprocessing
    2. Simulation-based inference (SBI)
    3. SRIM batch runs for sampled thetas
    4. Posterior predictive checks (PPC)
"""

import os
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import your modular utils package

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "data_path": "./data/all_vacancies.csv",
    "srim_dir": "/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013",       # path where TRIM.exe lives
    "results_base": "/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013/Outputs",
    "num_tracks": 10,                         # how many random x_obs (tracks) to sample
    "num_samples": 1000,                      # posterior samples per track
    "batch_size": 5,                          # memory safety for large runs
    "random_seed": 42,
}

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------


from src.utils.data_utils import preprocess

x_obs, theta, grouped = preprocess("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/all_vacancies.csv")

# -----------------------------------------------------------------------------
# Running SBI
# -----------------------------------------------------------------------------
import torch
from src.utils.sbi_runner import make_prior, make_inference, train_posterior

# Define prior, inference, and train posterior
prior = make_prior(low=[1000], high=[2000000])
inference = make_inference(prior, density_estimator='nsf')
posterior = train_posterior(inference, theta, x_obs)
print("TYPE OF POSTERIOR:", type(posterior))



# Save posterior 
# posterior_path = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/trained_posterior.pt"
# torch.save(posterior, posterior_path)
# posterior = torch.load(posterior_path, map_location="cpu")
# print(f"[INFO] Posterior saved to {posterior_path}")

# -----------------------------------------------------------------------------
# PPC
# -----------------------------------------------------------------------------
from src.utils.srim_utils import sample_posterior_bulk
from src.utils.srim_utils import pick_random_tracks
from src.utils.srim_utils import run_srim_multi_track
from src.utils.srim_parser import _find_file
from src.utils.srim_parser import _parse_tdata
from src.utils.srim_parser import _parse_vacancy
from src.utils.srim_parser import summarize_all_runs
from src.utils.data_utils import plot_ppc_histograms

x_test, track_ids = pick_random_tracks(x_obs, n=10)
samples_dict, _ = sample_posterior_bulk(posterior, x_test, track_ids=track_ids)
print("Available track_ids in samples_dict:", samples_dict.keys())
output_base = Path('/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013/Outputs')

run_srim_multi_track(
    samples_dict=samples_dict,
    x_test=x_test,
    track_ids=track_ids,
    srim_directory='/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013',
    output_base = output_base,
    ion_symbol="C",
    number_ions=50,
)

df_summary = summarize_all_runs('/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013/Outputs')

metrics = plot_ppc_histograms(
    df=df_summary,
    observed=x_obs,
    output_dir='/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/ppc-results',
    bins=40,        # smoother histograms
    save_plots=True,
    return_metrics=True
)

