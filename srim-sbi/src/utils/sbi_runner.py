import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform

def make_prior(low=[1000.0], high=[2_000_000.0]):
    """
    Create a uniform prior over energy range.

    Parameters
    ----------
    low : list[float]
        Lower bounds for parameters.
    high : list[float]s
        Upper bounds for parameters.

    Returns
    -------
    prior : torch.distributions.Distribution
        BoxUniform prior distribution.
    """
    low = torch.tensor(low)
    high = torch.tensor(high)
    prior = BoxUniform(low, high)
    return prior


def make_inference(prior, density_estimator="nsf"):
    """
    Initialize NPE inference object.

    Parameters
    ----------
    prior : torch.distributions.Distribution
        Prior distribution over parameters.
    density_estimator : str
        Type of neural density estimator ("maf", "nsf", etc.).

    Returns
    -------
    inference : sbi.inference.NPE
    """
    inference = NPE(prior=prior, density_estimator=density_estimator)
    return inference


def train_posterior(inference, theta, x_obs, epochs=None):
    """
    Train neural posterior estimator (NPE).

    Parameters
    ----------
    inference : sbi.inference.NPE
        NPE inference object (with prior and estimator).
    theta : torch.Tensor
        True parameters (e.g., energies).
    x_obs : torch.Tensor
        Observed data / features.
    epochs : int, optional
        Number of epochs to train for. Let SBI auto-determine if None.
    save_path : str, optional
        If provided, saves the trained posterior to disk.

    Returns
    -------
    posterior : sbi.inference.Posteriors
        Trained posterior object ready for sampling.
    """

    print("[Training NPE model...]")
    posterior_net = inference.append_simulations(theta, x_obs).train()
    print("[Training complete.]")

    posterior = inference.build_posterior(posterior_net)

    return posterior


def sample_posterior_thetas(posterior, x_test, num_samples=100):
    """
    Sample theta values from a trained posterior conditioned on test observations.

    Parameters
    ----------
    posterior : sbi.inference.posteriors.DirectPosterior
        Trained posterior object returned by `train_posterior`.
    x_test : torch.Tensor
        Test observation(s) to condition on. Shape (n_tracks, n_features).
    num_samples : int
        Number of posterior samples to draw per observation.

    Returns
    -------
    samples_dict : dict[int, torch.Tensor]
        Mapping from track index -> sampled theta values of shape (num_samples, n_params).
    """
    samples_dict = {}
    for i, x in enumerate(x_test):
        samples = posterior.sample((num_samples,), x=x)
        samples_dict[i] = samples
    return samples_dict


def sample_posterior_bulk(posterior, x_obs, num_samples=100, track_ids=None):
    """
    Sample posterior for multiple tracks and store results per track ID.

    Parameters
    ----------
    posterior : trained sbi posterior
        Trained posterior object (e.g., DirectPosterior)
    x_obs : torch.Tensor of shape (n_tracks, n_features)
        Observation tensor
    num_samples : int
        Number of posterior samples per track
    track_ids : list[int] | None
        Optional original indices for each observation in x_obs

    Returns
    -------
    samples_by_track : dict[int, torch.Tensor]
        Mapping track_id -> samples [num_samples, n_params]
    samples_tensor : torch.Tensor
        Combined samples [n_tracks, num_samples, n_params]
    """
    samples_by_track = {}
    all_tensors = []

    # Iterate over each observation and its associated track ID
    for i, x in enumerate(x_obs):
        with torch.no_grad():
            theta_samples = posterior.sample((num_samples,), x=x)

        # Determine the correct track ID
        track_id = track_ids[i] if track_ids is not None else str(i)

        samples_by_track[track_id] = theta_samples
        all_tensors.append(theta_samples)

    samples_tensor = torch.stack(all_tensors)
    return samples_by_track, samples_tensor
