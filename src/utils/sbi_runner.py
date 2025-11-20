# src/utils/sbi_runner.py

import torch
import io
import signal
import contextlib
from sbi.inference import NPE, MNPE
from sbi.utils import BoxUniform
import random
import math
import torch


# ============================================================
# MNPE: ENERGY + PARITY MODEL
# ============================================================

def make_mnpe_prior(energy_min=1.0, energy_max=100.0):
    """
    2-D prior over (energy, parity):
      energy ~ Uniform[energy_min, energy_max]
      parity ~ Uniform{0, 1}
    """
    low  = torch.tensor([energy_min, 0.0])
    high = torch.tensor([energy_max, 1.0])
    return BoxUniform(low, high)


def make_mnpe_inference(prior):
    """Create MNPE inference object."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MNPE(prior=prior, device=device)


def train_mnpe_posterior(inference, theta, x_obs):
    """
    Train MNPE posterior.
    - No normalization (prototype-style)
    - Uses direct sampling
    """
    print("[INFO] Training MNPE model...")
    inference.append_simulations(theta, x_obs).train()
    posterior = inference.build_posterior(sample_with="direct")
    print("[INFO] MNPE training complete.")
    return posterior


# ============================================================
# GUARDED MNPE POSTERIOR SAMPLING
# ============================================================

class TimeoutException(Exception):
    pass


def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """Run func with a hard timeout. Return None on timeout."""
    def handler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)

    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        print(f"[TIMEOUT] Exceeded {timeout_sec} sec → skipping.")
        return None
    finally:
        signal.alarm(0)

    return result


def guarded_posterior_sample(posterior, x, n_samples, hard_timeout_sec):
    """
    Call posterior.sample but guard against:
    - SBI "0% acceptance" failures
    - Hard timeouts

    Returns:
        Tensor or None
    """
    def run_sample():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            samples = posterior.sample((n_samples,), x=x)
        txt = buf.getvalue()

        if "Only 0.000% proposal samples" in txt:
            print("[GUARD] 0% acceptance detected → skipping.")
            return None

        return samples

    return run_with_timeout(run_sample, hard_timeout_sec)





def sample_energy(low, high, mode="continuous", step=None):
    """
    Energy sampler for random SRIM evaluation.

    mode = "continuous" → Uniform(low, high)
    mode = "grid"       → Uniform pick from grid with spacing 'step'
    mode = "biased_low" → More weights toward low energies
    """

    if mode == "continuous":
        return random.uniform(low, high)

    elif mode == "grid":
        if step is None:
            raise ValueError("step must be provided for grid sampling.")
        grid = torch.arange(low, high + 1e-9, step)
        idx = random.randrange(len(grid))
        return float(grid[idx])

    elif mode == "biased_low":
        # r^2 biases toward low energies strongly
        r = random.random()
        return low + (high - low) * (r ** 2)

    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
    



# ============================================================
# NPE: ENERGY-ONLY MODEL
# ============================================================

def make_npe_prior(low=[1000.0], high=[2_000_000.0]):
    """1-D prior for θ = energy (in eV or keV depending on your use)."""
    return BoxUniform(torch.tensor(low), torch.tensor(high))


def make_npe_inference(prior, density_estimator="nsf"):
    return NPE(prior=prior, density_estimator=density_estimator)


def train_npe_posterior(inference, theta, x_obs, epochs=None):
    """Train standard NPE model."""
    print("[Training NPE model...]")
    inference.append_simulations(theta, x_obs).train()
    posterior = inference.build_posterior()
    print("[Training complete.]")
    return posterior


# ============================================================
# NPE SAMPLING HELPERS
# ============================================================

def sample_npe_posterior_per_track(posterior, x_test, track_ids=None, n_samples=100):
    """Sample NPE posterior for each track independently."""
    out = {}
    for i, x in enumerate(x_test):
        with torch.no_grad():
            s = posterior.sample((n_samples,), x=x)
        tid = track_ids[i] if track_ids else str(i)
        out[tid] = s
    return out


def sample_npe_posterior_bulk(posterior, x_obs, n_samples=100, track_ids=None):
    """Sample NPE posterior for a whole dataset at once."""
    samples_by_track = {}
    stacked = []

    for i, x in enumerate(x_obs):
        with torch.no_grad():
            s = posterior.sample((n_samples,), x=x)
        tid = track_ids[i] if track_ids else str(i)
        samples_by_track[tid] = s
        stacked.append(s)

    return samples_by_track, torch.stack(stacked)