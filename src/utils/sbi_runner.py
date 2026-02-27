import io
import signal
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi.inference import NPE, MNPE, SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


# ============================================================
# 1. SHARED HELPERS (Timeouts & Sampling)
# ============================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def guarded_posterior_sample(posterior, x, n_samples, hard_timeout_sec=5):
    """
    Mac/Linux Compatible Timeout.
    Uses signal.SIGALRM to interrupt the sampling loop if it gets stuck.

    NOTE: A device-aware version of this function is defined further below,
    which shadows this one at import time. That version uses try/except
    instead of signal-based timeouts.
    """
    # Register the signal function handler
    signal.signal(signal.SIGALRM, timeout_handler)

    # Set the alarm (timer)
    signal.alarm(hard_timeout_sec)

    try:
        # Capture stdout to silence the "0.000% accepted" spam
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # The code will try to run this...
            samples = posterior.sample((n_samples,), x=x, show_progress_bars=False)

        # If successful, disable the alarm
        signal.alarm(0)
        return samples

    except TimeoutException:
        print(f"[WARN] Sampling TIMED OUT (> {hard_timeout_sec}s). Skipping track.")
        return None

    except Exception as e:
        # Catch other random crashes
        signal.alarm(0) # Ensure alarm is off
        print(f"[WARN] Sampling Error: {e}")
        return None

def make_prior(energy_min=1.0, energy_max=120.0):
    # Detect Device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    # Range for Energy: [1.0, 120.0]
    # Range for Class:  [0.0, 7.0] (Discrete indices 0-7)
    low  = torch.tensor([energy_min, 0.0], device=device)
    high = torch.tensor([energy_max, 7.0], device=device)
    return BoxUniform(low, high)

# =========================================================
# 2. PIPELINE A: SUMMARY FEATURES (MNPE)
# =========================================================

def make_inference_summary(prior):
    """
    Creates an MNPE instance for Summary Features.
    MNPE is designed for Mixed parameters (Continuous Energy + Discrete Angle).
    """
    # Detect Device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    # We use MNPE because our theta is [Energy (Cont), Class (Disc)]
    return MNPE(prior=prior, density_estimator="mnpe", device=device)


def train_summary_posterior(inference, theta, x):
    """
    Trains the Summary Feature model using MNPE.
    """
    print(f"[INFO] Training Summary Model (MNPE). Input Dim: {x.shape[1]}")
    print(f"[INFO] Theta Shape: {theta.shape} (Col 0=Energy, Col 1=Class)")

    # MNPE Training Loop
    inference.append_simulations(theta, x).train()

    # Build posterior with direct sampling
    posterior = inference.build_posterior(sample_with="direct")

    print("[INFO] MNPE Training Complete.")
    return posterior

# =========================================================
# 3. PIPELINE B: EMBEDDING NETWORK (NPE) — MDPE 4D PointNet
# =========================================================

class TNet(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)
        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(256)

        # THE FIX: Initialize weights to zero and bias to the flattened identity matrix.
        # This ensures the first forward pass is exactly the identity transformation.
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(input_dim).view(-1))

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Device Agnostic Identity Matrix
        iden = torch.eye(self.input_dim, requires_grad=True).repeat(bs, 1, 1).to(x.device)
        x = x.view(-1, self.input_dim, self.input_dim) + iden
        return x

class PointNetMulti(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=64):
        super().__init__()
        self.tnet = TNet(input_dim=input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.bn4 = nn.LayerNorm(256)
        self.bn5 = nn.LayerNorm(128)

    def forward(self, x):
        # Mask NaNs
        mask = ~torch.isnan(x[..., 0])
        # Prep Data: (Batch, Points, 3) -> (Batch, 3, Points)
        x_clean = torch.nan_to_num(x, nan=0.0).transpose(2, 1)

        # Alignment
        trans = self.tnet(x_clean)
        x_transformed = torch.bmm(trans, x_clean)

        # Features
        x = F.relu(self.bn1(self.conv1(x_transformed)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Apply Mask
        mask_expanded = mask.unsqueeze(1).float()
        x = x * mask_expanded

        # Multi-Pooling
        pool_max = torch.max(x, 2)[0]
        pool_sum = torch.sum(x, 2)
        valid_counts = torch.sum(mask_expanded, dim=2).clamp(min=1.0)
        pool_mean = pool_sum / valid_counts

        global_feat = torch.cat([pool_max, pool_sum, pool_mean], 1)

        # Final
        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

def make_embedding_architecture(latent_dim=64):
    return PointNetMulti(input_dim=4 , hidden_dim=128, output_dim=latent_dim)

def make_inference_embedding(prior, embedding_net):
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=embedding_net,
        z_score_x='none'
    )
    return NPE(prior=prior, density_estimator=neural_posterior, device=device)

def train_embedding_posterior(inference, theta, x_raw_tensor):
    print(f"[INFO] Training PointNetMulti Model (Input: {x_raw_tensor.shape})...")

    inference.append_simulations(
        theta,
        x_raw_tensor,
        exclude_invalid_x=False
    ).train()

    posterior = inference.build_posterior(sample_with="direct")
    print("[INFO] Training Complete.")
    return posterior

def make_inference_mnpe_embedding(prior, latent_dim=64):
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    embedding_net = PointNetMulti(input_dim=4, hidden_dim=128, output_dim=latent_dim)

    # THE FIX: Add z_score_x='none'
    density_estimator_fun = posterior_nn(
        model="mnpe",
        embedding_net=embedding_net,
        z_score_x='none'  # <--- THIS PREVENTS THE NaN ERROR
    )

    return MNPE(prior=prior, density_estimator=density_estimator_fun, device=device)

def train_mnpe_embedding_posterior(inference, theta, x_raw_tensor):
    """
    Trains the Mixed (Energy + Angle) model using Raw Point Clouds.
    """
    print(f"[INFO] Training MNPE with PointNet Embedding...")
    print(f"       Input Shape: {x_raw_tensor.shape} (Batch, Points, Feats)")
    print(f"       Theta Shape: {theta.shape} (Col 0=Energy, Col 1=Class)")

    # 1. Append Simulations
    # exclude_invalid_x=False allows handling padded data if your PointNet handles it (it does via mask)
    inference.append_simulations(theta, x_raw_tensor, exclude_invalid_x=False)

    # 2. Train
    inference.train()

    # 3. Build Posterior
    # MNPE generally requires direct sampling
    posterior = inference.build_posterior(sample_with="direct")

    print("[INFO] MNPE Embedding Training Complete.")
    return posterior


# ==========================================
# 4. MNPE HELPERS (SIIMPL / Rajendran)
# ==========================================

def get_device():
    """
    NOTE: This function is defined twice in this file (here and further below
    for the Rajendran comparison pipeline). The second definition shadows this one.
    Both return slightly different types (torch.device vs string).
    TODO: Consolidate into a single version.
    """
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def make_mnpe_prior(energy_min=100.0, energy_max=1000.0):
    device = get_device()
    # Range must cover 800.0 keV
    # Parity is 0 or 1. We set high=2.0 because BoxUniform is [low, high)
    low  = torch.tensor([energy_min, 0.0], device=device)
    high = torch.tensor([energy_max, 2.0], device=device)
    return BoxUniform(low, high)

def make_mnpe_inference(prior):
    device = get_device()
    # [cite_start]We use the standard MNPE class exactly as documented [cite: 4]
    return MNPE(prior=prior, device=str(device))

def train_mnpe_posterior(inference, theta, x):
    print("[INFO] Training MNPE model...")
    # [cite_start]Standard training call [cite: 31]
    inference.append_simulations(theta, x).train(
        training_batch_size=128,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20
    )

    print("[INFO] Building posterior...")
    posterior = inference.build_posterior(sample_with="direct")
    return posterior

def guarded_posterior_sample(posterior, x, n_samples=100, hard_timeout_sec=60):
    """
    Device-aware version of guarded_posterior_sample (shadows the signal-based version above).
    Uses simple try/except instead of signal-based timeouts.
    """
    try:
        # Strict device alignment to prevent crashes
        device = get_device()
        x = x.to(device)

        # Ensure posterior is on the same device if supported
        if hasattr(posterior, "to"):
            try: posterior.to(device)
            except: pass

        samples = posterior.sample((n_samples,), x=x, show_progress_bars=False)
        return samples.cpu()
    except Exception as e:
        print(f"[WARN] Sampling : {e}")
        return None


# =========================================================
# 5. CONTINUOUS ANGLE INFERENCE (MCPE)
# =========================================================

def setup_continuous_prior():
    """
    LOGIC: Define the boundaries for the AI's search.
    Target: [Energy, Sin(theta), Cos(theta)]
    """
    # Energy: 1 to 100 keV
    # Sin/Cos: Always bounded by [-1, 1]
    lower_bound = torch.tensor([1.0, -1.0, -1.0])
    upper_bound = torch.tensor([100.0, 1.0, 1.0])

    return BoxPrior(lower=lower_bound, upper=upper_bound)

def train_continuous_model(features, targets, prior):
    """
    LOGIC: Train the 'Brain' to map 26D features to 3D geometric coordinates.
    """
    # 1. Initialize SNPE (Sequential Neural Posterior Estimation)
    # This replaces the old 'Mixed' (MNPE) logic.
    inference = SNPE(prior=prior)

    # 2. Append the training data
    # features: (N, 26) tensor
    # targets: (N, 3) tensor [Energy, Sin, Cos]
    inference = inference.append_simulations(targets, features)

    # 3. Train the Normalizing Flow
    # The 'Flow' learns to warp a simple distribution into your track physics.
    density_estimator = inference.train()

    # 4. Build the Posterior
    posterior = inference.build_posterior(density_estimator)

    return posterior


# =========================================================
# 6. MCPE-3D: Pure 3D PointNet (Pipeline B)
# =========================================================

class TNet3D(nn.Module):
    """Calculates a 3x3 spatial rotation/alignment matrix."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # 3x3 matrix
        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(256)

        # Initialize to Identity Matrix
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(3).view(-1))

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, requires_grad=True).repeat(bs, 1, 1).to(x.device)
        x = x.view(-1, 3, 3) + iden
        return x

class PointNetContinuousPure3D(nn.Module):
    """Pure 3D PointNet strictly operating on (x, y, z) coordinates."""
    def __init__(self, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.tnet = TNet3D()

        # Convolutions now strictly take 3 features (x, y, z)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Max, Sum, Mean pooling = hidden_dim * 3
        self.fc1 = nn.Linear(hidden_dim * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim)

        self.bn4 = nn.LayerNorm(256)
        self.bn5 = nn.LayerNorm(128)

    def forward(self, x):
        # x shape: (Batch, Points, 3)

        # 1. Mask NaNs (Find valid points before zeroing them)
        mask = ~torch.isnan(x[..., 0])
        x_clean = torch.nan_to_num(x, nan=0.0) # Convert NaNs to 0.0 for math operations

        # 2. Apply TNet Alignment
        x_transposed = x_clean.transpose(2, 1) # (Batch, 3, Points)
        trans_matrix = self.tnet(x_transposed)
        x_aligned = torch.bmm(x_clean, trans_matrix) # (Batch, Points, 3)

        # Transpose back for 1D Convolutions
        x_aligned = x_aligned.transpose(2, 1) # (Batch, 3, Points)

        # 3. Extract Features
        feat = F.relu(self.bn1(self.conv1(x_aligned)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = F.relu(self.bn3(self.conv3(feat)))

        # 4. Apply Mask (Zero out features from padded points)
        mask_expanded = mask.unsqueeze(1).float()
        feat = feat * mask_expanded

        # 5. Multi-Pooling (Global Feature Extraction)
        pool_max = torch.max(feat, 2)[0]
        pool_sum = torch.sum(feat, 2)
        valid_counts = torch.sum(mask_expanded, dim=2).clamp(min=1.0)
        pool_mean = pool_sum / valid_counts

        global_feat = torch.cat([pool_max, pool_sum, pool_mean], 1)

        # 6. Output Latent Vector
        out = F.relu(self.bn4(self.fc1(global_feat)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out)

        return out

# =====================================================================
# SBI Wrapper Functions for 3D Embedding
# =====================================================================
def make_inference_3d_embedding(prior, latent_dim=64):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # Use the new pure 3D network
    embedding_net = PointNetContinuousPure3D(latent_dim=latent_dim)

    density_estimator_fun = posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
        z_score_x='none' # Bypasses SBI's normalizer so NaNs don't break it
    )

    return SNPE(prior=prior, density_estimator=density_estimator_fun, device=device)

def train_3d_embedding_posterior(inference, theta_tensor, x_raw_tensor):
    print(f"[INFO] Training 3D Continuous NPE with PointNet Embedding...")
    print(f"       Input Shape: {x_raw_tensor.shape} (Batch, Points, 3)")
    print(f"       Theta Shape: {theta_tensor.shape} [E, Vx, Vy, Vz]")

    inference.append_simulations(theta_tensor, x_raw_tensor, exclude_invalid_x=False)
    inference.train(show_train_summary=True)
    posterior = inference.build_posterior(sample_with="direct")

    print("[INFO] 3D Embedding Training Complete.")
    return posterior


# =========================================================
# 7. RAJENDRAN COMPARISON PIPELINE
# =========================================================

def get_device():
    """
    Returns device as a string (shadows the torch.device version above).
    TODO: Consolidate with the version above.
    """
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def make_mnpe_raj_prior(e_min=1.0, e_max=120.0):
    """
    Creates a prior for the Rajendran comparison range.
    Theta: [Energy (Continuous), Parity (Discrete 0 or 1)]
    """
    device = get_device()
    # Parity is 0 or 1. BoxUniform high is exclusive, so we use 2.0
    # to ensure 1.0 is fully covered in the prior space.
    low  = torch.tensor([e_min, 0.0], device=device)
    high = torch.tensor([e_max, 2.0], device=device)
    return BoxUniform(low, high)

def setup_mnpe_raj_inference(prior):
    """
    Sets up the MNPE inference engine specifically for
    mixed continuous/discrete parameters.
    """
    device = get_device()
    # MNPE is the specialized class for mixed types (e.g. Energy + Parity)
    return MNPE(prior=prior, device=device)

def train_mnpe_raj_posterior(inference, theta, x, training_batch_size=512, learning_rate=5e-4):
    """
    Trains the MNPE model.
    x: Summary features (e.g., the 14 features from preprocess_mnpe)
    theta: [Energy, Parity]
    """
    print(f"[INFO] Training MNPE-Raj Model.")
    print(f"       Features: {x.shape[1]} | Samples: {len(theta)}")

    # --- THE FIX: FORCING CONTINUOUS RECOGNITION ---
    # sbi auto-detects discrete parameters by looking for columns with very few unique values.
    # Because your dataset only has 8 fixed energies, MNPE thinks Energy is a discrete class!
    # By adding 0.1 eV of random noise, we create unique continuous values,
    # forcing the Neural Network to route Energy to the continuous Normalizing Flow.
    continuous_noise = torch.rand(theta.shape[0], device=theta.device) * 1e-4
    theta[:, 0] = theta[:, 0] + continuous_noise

    # Ensure parity is absolutely discrete (0.0 or 1.0)
    theta[:, 1] = torch.round(theta[:, 1])

    # Append simulations and train
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=training_batch_size,
        learning_rate=learning_rate,
        show_train_summary=True
    )

    # We use 'direct' sampling for MNPE as it is usually more efficient
    posterior = inference.build_posterior(density_estimator, sample_with="direct")

    print("[INFO] MNPE-Raj Training Complete.")
    return posterior


# =========================================================
# 8. LEGACY / OLD MNPE FUNCTIONS
# =========================================================

def make_mnpe_prior_old(energy_min=0.0, energy_max=3000.0):
    """
    Creates a joint prior for the MNPE model.
    Continuous first (Energy), Discrete second (Parity).
    """
    # Note: sbi usually prefers its own BoxUniform for continuous,
    # but for mixed types, a custom PyTorch distribution or None is often used
    # if you are just doing amortized inference on pre-generated data.
    # Returning None here is perfectly valid for basic MNPE training
    # as shown in the sbi documentation example.
    return None

def make_mnpe_inference_old(theta_train, x_train, prior=None, device='cpu'):
    """
    Initializes the MNPE model, appends the training data, and trains it.
    """

    continuous_noise = torch.rand(theta_train.shape[0]) * 0.5
    theta_train[:, 0] = theta_train[:, 0] + continuous_noise
    theta_train[:, 1] = torch.round(theta_train[:, 1])

    # Initialize the Mixed Neural Posterior Estimator [cite: 4, 24, 25]
    inference = MNPE(prior=prior, device=device)

    # Append the pre-simulated data and train the density estimator [cite: 26, 100, 101]
    # We use default training parameters, but you can override batch size or learning rate here [cite: 31]
    density_estimator = inference.append_simulations(theta_train, x_train).train(
        training_batch_size=200,
        validation_fraction=0.1,
        show_train_summary=True
    )

    return inference, density_estimator

def make_mnpe_posterior_old(inference, x_obs, num_samples=100):
    """
    Builds the posterior from the trained inference object and samples from it.
    """
    # Build the posterior object from the neural density estimator [cite: 28, 69]
    posterior = inference.build_posterior()

    # Ensure observation is at least 2D (1, num_features) [cite: 29]
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)

    # Sample from the posterior given the specific observation [cite: 30, 97]
    # Default uses 'direct' sampling [cite: 74]
    samples = posterior.sample((num_samples,), x=x_obs)

    return posterior, samples
