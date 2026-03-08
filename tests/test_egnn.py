"""
Equivariance Verification Tests for EGNN
=========================================
Run BEFORE building anything on top of the EGNN.
If these fail, the implementation has a bug and nothing downstream will work.

Usage:
    python -m pytest tests/test_egnn.py -v
"""

import torch
import numpy as np
import pytest
from scipy.stats import special_ortho_group

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.egnn import EGNNLayer, EGNNBackbone, EGNNEmbedding, build_knn_graph


def random_rotation_matrix():
    """Generate a random rotation matrix R in SO(3)."""
    R = torch.tensor(special_ortho_group.rvs(3), dtype=torch.float32)
    return R


def make_fake_track(n_points=50, n_max=100):
    """Create a fake track with known properties for testing."""
    # Random 3D point cloud (simulating a track)
    coords = torch.randn(1, n_points, 3)

    # Zero-pad to n_max
    padded = torch.zeros(1, n_max, 3)
    padded[0, :n_points, :] = coords

    mask = torch.zeros(1, n_max, dtype=torch.bool)
    mask[0, :n_points] = True

    return padded, mask


class TestEGNNLayerEquivariance:
    """Test that a single EGNN layer is equivariant."""

    def test_single_layer_equivariance(self):
        """
        Verify:
          1. h_out ≈ h_rot (scalar features unchanged under rotation)
          2. x_rot ≈ R @ x_out (positions rotate with input)
        """
        torch.manual_seed(42)
        d_h = 64
        layer = EGNNLayer(hidden_dim=d_h)
        layer.eval()

        # Create fake data
        N = 30  # number of points
        h = torch.randn(N, d_h)
        x = torch.randn(N, 3)

        # Simple kNN graph (fully connected for small N)
        src = []
        dst = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Forward pass on original
        with torch.no_grad():
            h_out, x_out = layer(h, x, edge_index)

        # Apply rotation to positions
        R = random_rotation_matrix()
        x_rot_input = (R @ x.T).T  # Rotate input positions

        # Forward pass on rotated input (same h, same edges)
        with torch.no_grad():
            h_rot, x_rot = layer(h, x_rot_input, edge_index)

        # Check 1: Scalar features should be identical
        assert torch.allclose(h_out, h_rot, atol=1e-5), \
            f"Scalar features changed under rotation! Max diff: {(h_out - h_rot).abs().max()}"

        # Check 2: Positions should rotate identically
        x_out_rotated = (R @ x_out.T).T
        assert torch.allclose(x_rot, x_out_rotated, atol=1e-5), \
            f"Positions not equivariant! Max diff: {(x_rot - x_out_rotated).abs().max()}"

        print("[PASS] Single EGNN layer is equivariant.")


class TestEGNNBackboneEquivariance:
    """Test that the full stacked backbone is equivariant."""

    def test_backbone_equivariance(self):
        torch.manual_seed(42)
        d_h = 64
        backbone = EGNNBackbone(hidden_dim=d_h, n_layers=4)
        backbone.eval()

        # Create fake track
        padded, mask = make_fake_track(n_points=40, n_max=60)

        # Build graph
        edge_index, batch_vec, flat_coords, real_counts, _ = build_knn_graph(
            padded, mask, k=10
        )

        # Forward pass on original
        with torch.no_grad():
            h_out, x_out = backbone(flat_coords, edge_index, real_counts, batch_vec)

        # Rotate input
        R = random_rotation_matrix()
        padded_rot = padded.clone()
        padded_rot[0, :40, :] = (R @ padded[0, :40, :].T).T

        edge_index_r, batch_r, flat_r, rc_r, _ = build_knn_graph(
            padded_rot, mask, k=10
        )

        with torch.no_grad():
            h_rot, x_rot = backbone(flat_r, edge_index_r, rc_r, batch_r)

        # Check scalar invariance
        assert torch.allclose(h_out, h_rot, atol=1e-4), \
            f"Backbone scalar features not invariant! Max diff: {(h_out - h_rot).abs().max()}"

        # Check position equivariance
        x_out_rotated = (R @ x_out.T).T
        assert torch.allclose(x_rot, x_out_rotated, atol=1e-4), \
            f"Backbone positions not equivariant! Max diff: {(x_rot - x_out_rotated).abs().max()}"

        print("[PASS] Full EGNN backbone (4 layers) is equivariant.")


class TestEGNNEmbeddingReadout:
    """Test that readout channels have correct properties."""

    def test_scalar_channel_invariance(self):
        """Scalar channel output should be IDENTICAL for original and rotated input."""
        torch.manual_seed(42)
        n_max = 60
        model = EGNNEmbedding(n_max=n_max, hidden_dim=32, n_layers=2, k=8,
                              n_heads=2, d_proj=16, d_latent=64)
        model.eval()

        padded, mask = make_fake_track(n_points=30, n_max=n_max)
        x_flat = padded.view(1, -1)  # (1, n_max*3)

        # Get scalar readout
        coords = padded
        edge_index, batch_vec, flat_coords, real_counts, _ = build_knn_graph(
            coords, mask, k=8
        )
        with torch.no_grad():
            h, x_new = model.backbone(flat_coords, edge_index, real_counts, batch_vec)
            e_scalar = model.scalar_readout(h, batch_vec, real_counts)

        # Rotate
        R = random_rotation_matrix()
        padded_rot = padded.clone()
        padded_rot[0, :30, :] = (R @ padded[0, :30, :].T).T

        edge_r, batch_r, flat_r, rc_r, _ = build_knn_graph(padded_rot, mask, k=8)
        with torch.no_grad():
            h_r, x_r = model.backbone(flat_r, edge_r, rc_r, batch_r)
            e_scalar_rot = model.scalar_readout(h_r, batch_r, rc_r)

        assert torch.allclose(e_scalar, e_scalar_rot, atol=1e-4), \
            f"Scalar channel not invariant! Max diff: {(e_scalar - e_scalar_rot).abs().max()}"

        print("[PASS] Scalar readout channel is rotation-invariant.")

    def test_full_forward_pass(self):
        """Test that the full model forward pass doesn't crash and returns correct shape."""
        torch.manual_seed(42)
        n_max = 60
        d_latent = 64
        model = EGNNEmbedding(n_max=n_max, hidden_dim=32, n_layers=2, k=8,
                              n_heads=2, d_proj=16, d_latent=d_latent)
        model.eval()

        padded, mask = make_fake_track(n_points=30, n_max=n_max)
        x_flat = padded.view(1, -1)  # (1, n_max*3)

        with torch.no_grad():
            z = model(x_flat)

        assert z.shape == (1, d_latent), f"Expected shape (1, {d_latent}), got {z.shape}"
        assert not torch.isnan(z).any(), "Output contains NaN!"
        assert not torch.isinf(z).any(), "Output contains Inf!"

        print(f"[PASS] Full forward pass produces shape {z.shape}, no NaN/Inf.")

    def test_batch_forward_pass(self):
        """Test with a batch of multiple tracks."""
        torch.manual_seed(42)
        n_max = 60
        d_latent = 64
        B = 4
        model = EGNNEmbedding(n_max=n_max, hidden_dim=32, n_layers=2, k=8,
                              n_heads=2, d_proj=16, d_latent=d_latent)
        model.eval()

        # Create batch of tracks with varying lengths
        x_batch = torch.zeros(B, n_max, 3)
        for i in range(B):
            n = 20 + i * 10  # 20, 30, 40, 50 points
            x_batch[i, :n, :] = torch.randn(n, 3)

        x_flat = x_batch.view(B, -1)

        with torch.no_grad():
            z = model(x_flat)

        assert z.shape == (B, d_latent), f"Expected shape ({B}, {d_latent}), got {z.shape}"
        assert not torch.isnan(z).any(), "Output contains NaN!"

        print(f"[PASS] Batch forward pass ({B} tracks) produces shape {z.shape}.")


if __name__ == "__main__":
    print("=" * 60)
    print("EGNN Equivariance Verification Tests")
    print("=" * 60)

    t1 = TestEGNNLayerEquivariance()
    t1.test_single_layer_equivariance()

    t2 = TestEGNNBackboneEquivariance()
    t2.test_backbone_equivariance()

    t3 = TestEGNNEmbeddingReadout()
    t3.test_scalar_channel_invariance()
    t3.test_full_forward_pass()
    t3.test_batch_forward_pass()

    print("\n" + "=" * 60)
    print("ALL EQUIVARIANCE TESTS PASSED")
    print("=" * 60)
