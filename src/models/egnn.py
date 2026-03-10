"""
EGNN-SBI Model: E(n)-Equivariant Graph Neural Network for SBI
=============================================================
Pipeline C: Replaces PointNet with an equivariant architecture that
naturally separates scalar features (rotation-invariant -> energy)
from vector features (rotation-equivariant -> direction).

Architecture:
  Stage 1: Preprocessing (center, PCA-align, normalize, pad, mask)
  Stage 2: EGNN Backbone (kNN graph + stacked EGNN layers)
  Stage 3: Dual-Channel Readout (scalar channel + vector channel + fusion)
  Stage 4: SNPE Normalizing Flow (external, SBI library)

Reference: Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# kNN Graph Builder (pure PyTorch, no torch_geometric needed)
# ============================================================

def build_knn_graph(coords, mask, k=16):
    """
    Build a kNN graph over real (non-padded) points only.
    Uses chunked processing: loops over small chunks of tracks,
    trimming each chunk to its max real length before computing cdist.
    This avoids the massive (B, N_max, N_max) distance matrix.

    Args:
        coords: (B, N, 3) padded coordinates
        mask:   (B, N)    boolean mask (True = real point)
        k:      number of nearest neighbors

    Returns:
        edge_index: (2, E) COO format edges (in flat/global indexing)
        batch_vec:  (total_real_points,) batch assignment
        flat_coords: (total_real_points, 3) real coordinates only
        real_counts: (B,) number of real points per track
        padded_to_flat: (B, N) mapping from padded to flat indices
    """
    B, N, _ = coords.shape
    device = coords.device
    CHUNK_SIZE = 16  # Process 16 tracks at a time

    real_counts = mask.sum(dim=1)  # (B,)

    # --- Extract flat coords and build batch_vec (vectorized) ---
    flat_coords = coords[mask]  # (total_real, 3)
    total_real = flat_coords.shape[0]

    batch_vec = mask.nonzero(as_tuple=True)[0]  # (total_real,)

    # padded_to_flat mapping
    padded_to_flat = torch.full((B, N), -1, dtype=torch.long, device=device)
    flat_indices = torch.arange(total_real, device=device)
    real_positions = mask.nonzero(as_tuple=False)  # (total_real, 2)
    padded_to_flat[real_positions[:, 0], real_positions[:, 1]] = flat_indices

    # Clamp k to max possible neighbors
    k_use = min(k, (real_counts.max().item() - 1)) if real_counts.max().item() > 1 else 0

    if k_use == 0 or total_real == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        return edge_index, batch_vec, flat_coords, real_counts, padded_to_flat

    # --- Chunked kNN: process CHUNK_SIZE tracks at a time ---
    edge_parts = []

    for chunk_start in range(0, B, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, B)
        chunk_coords = coords[chunk_start:chunk_end]     # (C, N, 3)
        chunk_mask = mask[chunk_start:chunk_end]           # (C, N)
        chunk_counts = real_counts[chunk_start:chunk_end]  # (C,)

        # Trim to max real points in this chunk (skip padding columns)
        max_real = int(chunk_counts.max().item())
        if max_real <= 1:
            continue
        chunk_coords = chunk_coords[:, :max_real]  # (C, M, 3) where M << N
        chunk_mask = chunk_mask[:, :max_real]        # (C, M)

        k_chunk = min(k_use, max_real - 1)
        C = chunk_end - chunk_start

        # Batched pairwise distances on the TRIMMED chunk
        dists = torch.cdist(chunk_coords, chunk_coords)  # (C, M, M)

        # Mask out padded points
        pad_mask = ~chunk_mask  # (C, M)
        dists.masked_fill_(pad_mask.unsqueeze(2), float('inf'))
        dists.masked_fill_(pad_mask.unsqueeze(1), float('inf'))

        # No self-loops
        diag_idx = torch.arange(max_real, device=device)
        dists[:, diag_idx, diag_idx] = float('inf')

        # Topk: (C, M, k_chunk)
        _, topk_local = dists.topk(k_chunk, dim=-1, largest=False)  # (C, M, k_chunk)

        # --- Vectorized conversion to flat global indices ---
        # Get the padded_to_flat mapping for this chunk (trimmed columns)
        chunk_p2f = padded_to_flat[chunk_start:chunk_end, :max_real]  # (C, M)

        # Build real-point mask for this chunk: which (b_local, n) are real
        # chunk_mask is (C, M), True for real points
        chunk_real_positions = chunk_mask.nonzero(as_tuple=False)  # (num_real_in_chunk, 2)
        if chunk_real_positions.shape[0] == 0:
            continue

        b_local = chunk_real_positions[:, 0]  # (num_real,)
        n_local = chunk_real_positions[:, 1]  # (num_real,)

        # Source flat indices for real points in this chunk
        src_flat = chunk_p2f[b_local, n_local]  # (num_real,)

        # Gather topk neighbor indices for real points: (num_real, k_chunk)
        topk_for_real = topk_local[b_local, n_local, :]  # (num_real, k_chunk)

        # Map neighbor padded indices to flat indices
        dst_flat = chunk_p2f[
            b_local.unsqueeze(1).expand_as(topk_for_real),
            topk_for_real
        ]  # (num_real, k_chunk)

        # Expand src to match: (num_real, k_chunk)
        src_expanded = src_flat.unsqueeze(1).expand_as(dst_flat)

        # Flatten
        src_1d = src_expanded.reshape(-1)
        dst_1d = dst_flat.reshape(-1)

        # Remove invalid edges (padded neighbors mapped to -1)
        valid = (dst_1d >= 0) & (src_1d >= 0)
        if valid.any():
            edge_parts.append(torch.stack([src_1d[valid], dst_1d[valid]], dim=0))

    if len(edge_parts) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    else:
        edge_index = torch.cat(edge_parts, dim=1)  # (2, E)

    return edge_index, batch_vec, flat_coords, real_counts, padded_to_flat


def build_edges_precomputed(coords, mask, neighbor_idx):
    """
    Build edge_index from PRECOMPUTED kNN neighbor indices.
    No distance computation — just index mapping. ~100x faster than cdist.

    Args:
        coords:       (B, N, 3) padded coordinates
        mask:         (B, N)    boolean mask (True = real point)
        neighbor_idx: (B, N, k) precomputed kNN indices in padded space
                      (-1 for padding / invalid neighbors)

    Returns:
        edge_index:     (2, E) COO format edges (flat/global indexing)
        batch_vec:      (total_real,) batch assignment
        flat_coords:    (total_real, 3) real coordinates only
        real_counts:    (B,) number of real points per track
        padded_to_flat: (B, N) mapping from padded to flat indices
    """
    B, N, _ = coords.shape
    device = coords.device

    real_counts = mask.sum(dim=1)  # (B,)

    # Extract flat coords and batch_vec
    flat_coords = coords[mask]      # (total_real, 3)
    total_real = flat_coords.shape[0]
    batch_vec = mask.nonzero(as_tuple=True)[0]  # (total_real,)

    # padded_to_flat mapping
    padded_to_flat = torch.full((B, N), -1, dtype=torch.long, device=device)
    flat_indices = torch.arange(total_real, device=device)
    real_positions = mask.nonzero(as_tuple=False)  # (total_real, 2)
    padded_to_flat[real_positions[:, 0], real_positions[:, 1]] = flat_indices

    if total_real == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        return edge_index, batch_vec, flat_coords, real_counts, padded_to_flat

    # Get (b, n) for each real point
    b_idx = real_positions[:, 0]  # (total_real,)
    n_idx = real_positions[:, 1]  # (total_real,)

    # Gather precomputed neighbors for real points: (total_real, k)
    k = neighbor_idx.shape[2]
    nbr_for_real = neighbor_idx[b_idx, n_idx, :]  # (total_real, k)

    # Map neighbor padded indices to flat global indices
    # Clamp to valid range for indexing (fix -1 after)
    nbr_clamped = nbr_for_real.clamp(min=0)
    dst_flat = padded_to_flat[
        b_idx.unsqueeze(1).expand_as(nbr_clamped),
        nbr_clamped
    ]  # (total_real, k)
    # Restore -1 for invalid neighbors
    dst_flat[nbr_for_real < 0] = -1

    # Source: each real point repeated k times
    src_flat = flat_indices.unsqueeze(1).expand_as(dst_flat)  # (total_real, k)

    # Flatten and filter invalid
    src_1d = src_flat.reshape(-1)
    dst_1d = dst_flat.reshape(-1)
    valid = dst_1d >= 0
    edge_index = torch.stack([src_1d[valid], dst_1d[valid]], dim=0)  # (2, E)

    return edge_index, batch_vec, flat_coords, real_counts, padded_to_flat


# ============================================================
# Single EGNN Layer
# ============================================================

class EGNNLayer(nn.Module):
    """
    One layer of the E(n)-Equivariant Graph Neural Network.

    Updates two streams simultaneously:
      - h (scalar features): rotation-invariant, with residual connection
      - x (positions):       rotation-equivariant, NO residual
    """

    def __init__(self, hidden_dim):
        super().__init__()
        d_h = hidden_dim

        # Edge MLP: phi_e([h_i, h_j, d_ij]) -> message
        self.phi_e = nn.Sequential(
            nn.Linear(2 * d_h + 1, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )

        # Coord MLP: phi_x(message) -> scalar weight for position update
        self.phi_x = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.SiLU(),
            nn.Linear(d_h, 1),
        )

        # Node MLP: phi_h([h_i, aggregated_messages]) -> updated features
        self.phi_h = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(self, h, x, edge_index):
        """
        Args:
            h: (N, d_h) scalar node features
            x: (N, 3)   node positions
            edge_index: (2, E) graph edges [sources, destinations]

        Returns:
            h_new: (N, d_h) updated scalar features (with residual)
            x_new: (N, 3)   updated positions (no residual)
        """
        src, dst = edge_index  # src -> dst edges

        # 1. Compute edge messages
        d_ij = (x[src] - x[dst]).pow(2).sum(dim=-1, keepdim=True)  # (E, 1) squared distance
        edge_input = torch.cat([h[src], h[dst], d_ij], dim=-1)     # (E, 2*d_h + 1)
        m_ij = self.phi_e(edge_input)                                # (E, d_h)

        # 2. Position update (equivariant)
        w_ij = torch.tanh(self.phi_x(m_ij))     # (E, 1) scalar weight
        x_diff = x[src] - x[dst]                 # (E, 3) direction vectors
        weighted_diff = w_ij * x_diff             # (E, 3)

        # Aggregate weighted diffs per node, normalized by degree
        N = h.shape[0]
        agg_x = torch.zeros(N, 3, device=x.device, dtype=x.dtype)
        agg_x.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_diff), weighted_diff)

        # Normalize by neighbor count
        degree = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        degree.scatter_add_(0, dst.unsqueeze(-1), torch.ones(src.shape[0], 1, device=x.device))
        degree = degree.clamp(min=1)

        x_new = x + agg_x / degree  # (N, 3)

        # 3. Scalar feature update (invariant) with residual
        agg_m = torch.zeros(N, m_ij.shape[1], device=h.device, dtype=h.dtype)
        agg_m.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)

        node_input = torch.cat([h, agg_m], dim=-1)  # (N, 2*d_h)
        h_new = self.phi_h(node_input) + h           # Residual connection

        return h_new, x_new


# ============================================================
# EGNN Backbone (stacked layers)
# ============================================================

class EGNNBackbone(nn.Module):
    """
    Stack of EGNN layers with shared kNN graph.

    Initializes scalar features h to 1 for real points, 0 for padding.
    """

    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Initial feature projection (from 1-dim indicator to hidden_dim)
        self.h_init = nn.Linear(1, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, flat_coords, edge_index, real_counts, batch_vec):
        """
        Args:
            flat_coords: (total_real, 3) coordinates of real points only
            edge_index:  (2, E) kNN graph edges
            real_counts: (B,) number of real points per track
            batch_vec:   (total_real,) batch assignment

        Returns:
            h_final: (total_real, d_h) scalar features per point
            x_final: (total_real, 3) updated positions per point
        """
        N = flat_coords.shape[0]

        # Initialize h: all ones for real points (indicator feature)
        h_indicator = torch.ones(N, 1, device=flat_coords.device)
        h = self.h_init(h_indicator)  # (N, d_h)

        x = flat_coords.clone()

        # Stack EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        return h, x


# ============================================================
# Dual-Channel Readout
# ============================================================

class ScalarReadout(nn.Module):
    """
    Channel A: Rotation-invariant readout for energy + morphology.
    Mean pool + max pool over real points, concatenated.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.output_dim = 2 * hidden_dim

    def forward(self, h, batch_vec, real_counts):
        """
        Args:
            h: (total_real, d_h) per-point scalar features
            batch_vec: (total_real,) batch assignment
            real_counts: (B,) real point counts

        Returns:
            e_scalar: (B, 2*d_h) concatenated mean and max pool
        """
        B = real_counts.shape[0]
        d_h = h.shape[1]
        device = h.device

        # Mean pool
        h_sum = torch.zeros(B, d_h, device=device, dtype=h.dtype)
        h_sum.scatter_add_(0, batch_vec.unsqueeze(-1).expand_as(h), h)
        h_mean = h_sum / real_counts.unsqueeze(-1).float().clamp(min=1)

        # Max pool (initialize with -inf)
        h_max = torch.full((B, d_h), float('-inf'), device=device, dtype=h.dtype)
        h_max.scatter_reduce_(0, batch_vec.unsqueeze(-1).expand_as(h), h, reduce='amax')

        return torch.cat([h_mean, h_max], dim=-1)  # (B, 2*d_h)


class VectorReadout(nn.Module):
    """
    Channel B: Rotation-equivariant readout for direction.

    K attention heads, each producing a weighted sum of positions,
    projected to higher dimension.
    """

    def __init__(self, hidden_dim, n_heads=4, d_proj=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_proj = d_proj
        self.output_dim = n_heads * d_proj

        # Attention MLPs (one per head)
        self.attn_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            for _ in range(n_heads)
        ])

        # Projection matrices (3D vector -> d_proj)
        self.projections = nn.ModuleList([
            nn.Linear(3, d_proj, bias=False)
            for _ in range(n_heads)
        ])

    def forward(self, h, x, batch_vec, real_counts):
        """
        Args:
            h: (total_real, d_h) per-point scalar features (invariant)
            x: (total_real, 3)   per-point positions (equivariant)
            batch_vec: (total_real,) batch assignment
            real_counts: (B,)

        Returns:
            e_vector: (B, K * d_proj) concatenated projected attention vectors
        """
        B = real_counts.shape[0]
        device = h.device
        head_outputs = []

        for k in range(self.n_heads):
            # Compute attention scores from invariant features
            alpha = self.attn_mlps[k](h).squeeze(-1)  # (total_real,)

            # Masked softmax per track
            # Set padding scores to -inf (but we only have real points here)
            # We need per-batch softmax
            alpha_exp = torch.exp(alpha - self._scatter_max(alpha, batch_vec, B))
            alpha_sum = torch.zeros(B, device=device, dtype=alpha_exp.dtype)
            alpha_sum.scatter_add_(0, batch_vec, alpha_exp)
            weights = alpha_exp / alpha_sum[batch_vec].clamp(min=1e-8)  # (total_real,)

            # Weighted sum of positions
            weighted_x = weights.unsqueeze(-1) * x  # (total_real, 3)
            v_k = torch.zeros(B, 3, device=device, dtype=x.dtype)
            v_k.scatter_add_(0, batch_vec.unsqueeze(-1).expand_as(weighted_x), weighted_x)
            # v_k is (B, 3) — equivariant!

            # Project to higher dimension
            p_k = self.projections[k](v_k)  # (B, d_proj)
            head_outputs.append(p_k)

        return torch.cat(head_outputs, dim=-1)  # (B, K * d_proj)

    def _scatter_max(self, src, index, num_groups):
        """Compute per-group max for numerically stable softmax."""
        out = torch.full((num_groups,), float('-inf'), device=src.device, dtype=src.dtype)
        out.scatter_reduce_(0, index, src, reduce='amax')
        return out[index]


class FusionMLP(nn.Module):
    """Fuse scalar and vector channels into a fixed-size latent embedding."""

    def __init__(self, scalar_dim, vector_dim, d_latent=256):
        super().__init__()
        total_dim = scalar_dim + vector_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, d_latent),
        )

    def forward(self, e_scalar, e_vector):
        e_full = torch.cat([e_scalar, e_vector], dim=-1)
        return self.mlp(e_full)


# ============================================================
# Full EGNN Embedding Network (for SBI)
# ============================================================

class EGNNEmbedding(nn.Module):
    """
    Complete EGNN embedding network that plugs into SBI's SNPE.

    Takes a flattened padded tensor (B, N_max * 3) from SBI,
    reshapes it, builds kNN graph, runs EGNN + readout, and
    returns a fixed-size embedding (B, d_latent).

    Args:
        n_max:      maximum number of points per track (padding size)
        hidden_dim: EGNN hidden feature dimension
        n_layers:   number of EGNN layers
        k:          number of nearest neighbors for graph
        n_heads:    number of attention heads in vector readout
        d_proj:     projection dimension per head
        d_latent:   final embedding dimension
    """

    def __init__(
        self,
        n_max,
        hidden_dim=64,
        n_layers=4,
        k=16,
        n_heads=4,
        d_proj=64,
        d_latent=256,
    ):
        super().__init__()
        self.n_max = n_max
        self.k = k

        # Backbone
        self.backbone = EGNNBackbone(hidden_dim=hidden_dim, n_layers=n_layers)

        # Readout
        self.scalar_readout = ScalarReadout(hidden_dim)
        self.vector_readout = VectorReadout(hidden_dim, n_heads=n_heads, d_proj=d_proj)
        self.fusion = FusionMLP(
            scalar_dim=self.scalar_readout.output_dim,
            vector_dim=self.vector_readout.output_dim,
            d_latent=d_latent,
        )

    def forward(self, x_flat):
        """
        Args:
            x_flat: Either:
              - (B, N_max * 3) coords only — computes kNN on the fly (slow, for tests)
              - (B, N_max * (3 + k)) coords + precomputed kNN neighbors (fast, for training)

        Returns:
            z: (B, d_latent) fixed-size embedding for the normalizing flow
        """
        B = x_flat.shape[0]
        device = x_flat.device
        D = x_flat.shape[1]

        coords_only_dim = self.n_max * 3
        precomputed_dim = self.n_max * (3 + self.k)

        if D == precomputed_dim:
            # --- FAST PATH: precomputed kNN neighbors embedded in x_flat ---
            coords_flat = x_flat[:, :coords_only_dim]
            nbr_flat = x_flat[:, coords_only_dim:]

            coords = coords_flat.view(B, self.n_max, 3)
            neighbor_idx = nbr_flat.view(B, self.n_max, self.k).long()

            # Mask: padding slots are exactly 0.0 in all 3 coords.
            # Use exact-zero check so real points near the centroid aren't
            # incorrectly masked out (the old threshold 1e-8 could do that).
            mask = coords.abs().sum(dim=-1) > 0.0  # (B, N_max)

            edge_index, batch_vec, flat_coords, real_counts, _ = build_edges_precomputed(
                coords, mask, neighbor_idx
            )
        else:
            # --- SLOW PATH: compute kNN on the fly (for tests / backward compat) ---
            coords = x_flat.view(B, self.n_max, 3)
            mask = coords.abs().sum(dim=-1) > 0.0

            edge_index, batch_vec, flat_coords, real_counts, _ = build_knn_graph(
                coords, mask, k=self.k
            )

        # Run EGNN backbone
        h_final, x_final = self.backbone(flat_coords, edge_index, real_counts, batch_vec)

        # Dual-channel readout
        e_scalar = self.scalar_readout(h_final, batch_vec, real_counts)
        e_vector = self.vector_readout(h_final, x_final, batch_vec, real_counts)

        # Fusion
        z = self.fusion(e_scalar, e_vector)

        return z  # (B, d_latent)
