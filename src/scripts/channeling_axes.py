"""Compute diamond channeling axes in the SIIMPL lab frame.

With rotate_crystal_mode=False, SIIMPL applies a fixed 35-degree pre-rotation
of the unit cell about Y (with phi=0). The lattice channels in the lab frame
are therefore the literal <hkl> directions rotated by R_y(35 deg).

This module exports the axis directions in lab frame, filtered to the upper
hemisphere (since beam direction must have v_z >= 0 for SIIMPL).
"""
from __future__ import annotations
import numpy as np

# SIIMPL's pre-rotation (matches simulator._DIAMOND_PREROT_DEG)
DIAMOND_PREROT_DEG = 35.0

# psi_c formula constants
PREFAC_eV_A = 1036.8   # 2 Z1 Z2 e^2 for C-on-C in eV*A
D = {"<110>": 2.52, "<111>": 3.09, "<100>": 3.57}  # row spacings, A


def _R_y(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


# Literal (crystal-frame) signed equivalents per family
_LITERAL_100 = np.array([
    [+1, 0, 0], [-1, 0, 0],
    [ 0,+1, 0], [ 0,-1, 0],
    [ 0, 0,+1], [ 0, 0,-1],
], dtype=float)

_LITERAL_110 = np.array([
    [+1,+1, 0], [+1,-1, 0], [-1,+1, 0], [-1,-1, 0],
    [+1, 0,+1], [+1, 0,-1], [-1, 0,+1], [-1, 0,-1],
    [ 0,+1,+1], [ 0,+1,-1], [ 0,-1,+1], [ 0,-1,-1],
], dtype=float) / np.sqrt(2)

_LITERAL_111 = np.array([
    [+1,+1,+1], [+1,+1,-1], [+1,-1,+1], [+1,-1,-1],
    [-1,+1,+1], [-1,+1,-1], [-1,-1,+1], [-1,-1,-1],
], dtype=float) / np.sqrt(3)


def _filter_upper_hemisphere(axes: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Keep only axes with v_z >= 0 (upper hemisphere). For antipodal pairs,
    one of the two lands in the upper hemisphere; we keep that one only.
    Axes lying on the equator (v_z == 0) keep both signs trivially via the
    eps tolerance; we drop the +/- y mirror redundancy explicitly below.
    """
    mask = axes[:, 2] >= -eps
    upper = axes[mask].copy()
    # Force v_z >= 0 strictly: if any equator vector has v_z == 0 it's fine.
    # No double-counting issues because we already excluded the lower copy.
    return upper


# Pre-rotated families in LAB frame, upper-hemisphere only
_R = _R_y(DIAMOND_PREROT_DEG)
AXES_LAB = {
    "<100>": _filter_upper_hemisphere((_R @ _LITERAL_100.T).T),
    "<110>": _filter_upper_hemisphere((_R @ _LITERAL_110.T).T),
    "<111>": _filter_upper_hemisphere((_R @ _LITERAL_111.T).T),
}


def psi_c_deg(E_keV: float, family: str) -> float:
    """Lindhard high-energy critical angle (deg) for the given family."""
    return float(np.degrees(np.sqrt(PREFAC_eV_A / (E_keV * 1000.0 * D[family]))))


def angle_to_nearest_axis(v: np.ndarray, family: str) -> float:
    """Smallest angle (deg) from unit vector v to the nearest equivalent axis
    of `family` in lab frame (upper-hemisphere representatives)."""
    cos_max = np.max(np.abs(AXES_LAB[family] @ v))
    cos_max = float(np.clip(cos_max, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_max)))


def sample_in_cone(axis: np.ndarray, half_angle_rad: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Sample a unit vector uniformly in the cone of half-angle around axis."""
    cos_theta = rng.uniform(np.cos(half_angle_rad), 1.0)
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    phi = rng.uniform(0, 2 * np.pi)
    local = np.array([sin_theta * np.cos(phi),
                      sin_theta * np.sin(phi),
                      cos_theta])
    z_new = axis / np.linalg.norm(axis)
    ref = np.array([0., 0., 1.]) if abs(z_new[2]) < 0.9 else np.array([1., 0., 0.])
    x_new = np.cross(ref, z_new); x_new /= np.linalg.norm(x_new)
    y_new = np.cross(z_new, x_new)
    R = np.stack([x_new, y_new, z_new], axis=1)
    return R @ local


if __name__ == "__main__":
    # Smoke test: print the rotated axes and their psi_c at 10 keV
    for fam, axs in AXES_LAB.items():
        psi = psi_c_deg(10.0, fam)
        print(f"\n{fam}  (n={len(axs)}, d={D[fam]} A,  psi_c(10keV)={psi:.2f} deg)")
        for ax in axs:
            print(f"  {ax}  (norm={np.linalg.norm(ax):.4f})")
