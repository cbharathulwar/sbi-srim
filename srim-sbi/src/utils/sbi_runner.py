import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform


def make_prior(low=[1000.0], high=[2_000_000.0]):
    """
    Make a uniform prior over the energy range.
    Returns a torch BoxUniform distribution.
    """
    low = torch.tensor(low)
    high = torch.tensor(high)
    return BoxUniform(low, high)



def make_inference(prior, density_estimator="nsf"):
    """
    Set up the neural posterior estimator (NPE) for inference.
    """
    return NPE(prior=prior, density_estimator=density_estimator)

def train_posterior(inference, theta, x_obs, epochs=None):
    print("[Training NPE model...]")
    posterior_net = inference.append_simulations(theta, x_obs).train()
    print("[Training complete.]")

    posterior = inference.build_posterior(posterior_net)
    return posterior

    return inference.build_posterior(posterior_net)

def sample_posterior_thetas(posterior, x_test, track_ids=None, num_samples=100):
    """
    Draw samples from the trained posterior for each test track.
    """
    samples_dict = {}
    for i, x in enumerate(x_test):
        with torch.no_grad():
            samples = posterior.sample((num_samples,), x=x)
        tid = track_ids[i] if track_ids is not None else str(i)
        samples_dict[tid] = samples
    return samples_dict


def sample_posterior_bulk(posterior, x_obs, num_samples=100, track_ids=None):
    """
    Sample posterior for multiple tracks and store results per track ID.

    Parameters
    ----------
    posterior : sbi.inference.posteriors.DirectPosterior
        Trained posterior object (e.g., DirectPosterior)
    x_obs : torch.Tensor of shape (n_tracks, n_features)
        Observation tensor
    num_samples : int
        Number of posterior samples per track
    track_ids : list[str] | None
        Optional readable track IDs corresponding to x_obs.

    Returns
    -------
    samples_by_track : dict[str, torch.Tensor]
        Mapping track_id -> samples [num_samples, n_params]
    samples_tensor : torch.Tensor
        Combined samples [n_tracks, num_samples, n_params]
    """
    samples_by_track = {}
    all_tensors = []

    for i, x in enumerate(x_obs):
        with torch.no_grad():
            theta_samples = posterior.sample((num_samples,), x=x)

        track_id = track_ids[i] if track_ids is not None else str(i)

        samples_by_track[track_id] = theta_samples
        all_tensors.append(theta_samples)

    samples_tensor = torch.stack(all_tensors)
    return samples_by_track, samples_tensor
