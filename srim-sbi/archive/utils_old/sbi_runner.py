# src/utils/sbi_runner.py

import torch
from sbi.inference import NPE, MNPE
from sbi.utils import BoxUniform

def normalize_energy(theta):
    """
    Normalize only the energy (column 0 of theta).
    Returns (theta_normed, stats_dict)
    """
    E = theta[:, 0]
    E_mean = E.mean()
    E_std = E.std()
    theta_normed = theta.clone()
    theta_normed[:, 0] = (E - E_mean) / E_std

    stats = {"E_mean": E_mean.item(), "E_std": E_std.item()}
    print(f"[INFO] Energy normalized: mean={E_mean:.3f}, std={E_std:.3f}")
    return theta_normed, stats


def make_mnpe_prior2(energy_min=1.0, energy_max=100.0):
    """
    Prior over continuous energy (keV).
    Parity is handled internally by MNPE as discrete.
    """
    low = torch.tensor([energy_min])
    high = torch.tensor([energy_max])
    return BoxUniform(low, high)


def make_mnpe_prior(energy_min=1.0, energy_max=100.0):
    """
    2-D prior over (energy, parity):
      energy ~ Uniform[energy_min, energy_max]
      parity ~ Uniform[0, 1]
    """
    low = torch.tensor([energy_min, 0.0])
    high = torch.tensor([energy_max, 1.0])
    return BoxUniform(low, high)


def make_mnpe_inference(prior):
    """
    Create MNPE inference object: θ = (energy, parity).
    """
    inference = MNPE(
        prior=prior,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("[INFO] Created MNPE inference object.")
    return inference


def train_mnpe_posterior(inference, theta, x_obs):
    """
    Train MNPE posterior — prototype-style:
    - No θ normalization
    - No x normalization
    - Direct sampling posterior
    """
    print("[INFO] Training MNPE model...")
    posterior_net = inference.append_simulations(theta, x_obs).train()
    # PROTOTYPE-STYLE: use direct sampling
    posterior = inference.build_posterior(sample_with="direct")
    print("[INFO] MNPE training complete.")
    return posterior


# ---------------------------
# NPE: ENERGY-ONLY MODEL
# ---------------------------

def make_npe_prior(low=[1000.0], high=[2_000_000.0]):
    return BoxUniform(torch.tensor(low), torch.tensor(high))


def make_npe_inference(prior, density_estimator="nsf"):
    return NPE(prior=prior, density_estimator=density_estimator)


def train_npe_posterior(inference, theta, x_obs, epochs=None):
    print("[Training NPE model...]")
    posterior_net = inference.append_simulations(theta, x_obs).train()
    posterior = inference.build_posterior(posterior_net)
    print("[Training complete.]")
    return posterior


def sample_npe_posterior_per_track(posterior, x_test, track_ids=None, n_samples=100):
    samples_dict = {}
    for i, x in enumerate(x_test):
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=x)
        tid = track_ids[i] if track_ids is not None else str(i)
        samples_dict[tid] = samples
    return samples_dict


def sample_npe_posterior_bulk(posterior, x_obs, n_samples=100, track_ids=None):
    samples_by_track = {}
    stacked = []
    for i, x in enumerate(x_obs):
        with torch.no_grad():
            s = posterior.sample((n_samples,), x=x)
        tid = track_ids[i] if track_ids is not None else str(i)
        samples_by_track[tid] = s
        stacked.append(s)
    return samples_by_track, torch.stack(stacked)