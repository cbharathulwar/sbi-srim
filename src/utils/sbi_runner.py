import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import signal
import contextlib
# --- CRITICAL: Explicitly import MNPE ---
from sbi.inference import NPE, MNPE 
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

# ============================================================
# 1. SHARED HELPERS (Timeouts & Sampling)
# ============================================================

class TimeoutException(Exception): pass

def run_with_timeout(func, timeout_sec, *args, **kwargs):
    def handler(signum, frame):
        raise TimeoutException()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)
    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        return None
    finally:
        signal.alarm(0)
    return result

def guarded_posterior_sample(posterior, x, n_samples, hard_timeout_sec):
    """
    Call posterior.sample but guard against timeouts.
    """
    def run_sample():
        # Capturing stdout to suppress internal SBI prints if needed
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            samples = posterior.sample((n_samples,), x=x)
        return samples

    return run_with_timeout(run_sample, hard_timeout_sec)

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
# 3. PIPELINE B: EMBEDDING NETWORK (NPE)
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



    return MNPE(prior=prior, density_estimator=density_estimator_fun, device=device)

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


import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

import torch
from sbi.inference import MNPE  # <--- Using the specialized class directly
from sbi.utils import BoxUniform
import torch
from sbi.inference import MNPE
from sbi.utils import BoxUniform

import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn  # <--- The key to manual configuration
import torch
from sbi.inference import MNPE
from sbi.utils import BoxUniform
import torch
from sbi.inference import MNPE
from sbi.utils import BoxUniform


import torch
from sbi.inference import MNPE
from sbi.utils import BoxUniform

# ==========================================
# 0. DEVICE HELPER
# ==========================================
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

# ==========================================
# 1. PRIOR
# ==========================================
def make_mnpe_prior(energy_min=100.0, energy_max=1000.0):
    device = get_device()
    # Range must cover 800.0 keV
    # Parity is 0 or 1. We set high=2.0 because BoxUniform is [low, high)
    low  = torch.tensor([energy_min, 0.0], device=device)
    high = torch.tensor([energy_max, 2.0], device=device)
    return BoxUniform(low, high)

# ==========================================
# 2. INFERENCE (Standard API)
# ==========================================
def make_mnpe_inference(prior):
    device = get_device()
    # [cite_start]We use the standard MNPE class exactly as documented [cite: 4]
    return MNPE(prior=prior, device=str(device))

# ==========================================
# 3. TRAIN
# ==========================================
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

# ==========================================
# 4. SAMPLER
# ==========================================
def guarded_posterior_sample(posterior, x, n_samples=100, hard_timeout_sec=60):
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