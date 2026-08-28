"""
Reconstruct an object's shape from MANY sparse, noisy, per-frame depth clouds
(already in world coordinates).

The problem: each frame's object depth is sparse, non-uniform, and has blur
fliers. The insight: you have many such views in one shared world frame, so
multi-view AGGREGATION is the densifier and CROSS-VIEW AGREEMENT is the
denoiser -- no depth-completion network, no loss of metric scale.

Pipeline (all sparse-point operations, no meshing, no learning):
  A. per-frame filter   -- drop isolated fliers before they pollute the union
  B. fuse               -- concatenate all frames (shared world frame)
  C. consistency filter -- keep points supported by enough neighbours (real
                           surface is re-observed across views; noise isn't)
  D. uniform resample   -- voxel-average to one point per cell -> even density,
                           shape preserved, memory bounded

Input : list of (N_i, 3) world-frame clouds (one per view)
Output: (M, 3) clean, uniformly-sampled cloud holding the object's shape
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def radius_outlier_filter(xyz: np.ndarray, radius: float,
                          min_neighbors: int) -> np.ndarray:
    """
    Keep points that have >= min_neighbors OTHER points within `radius`.
    Removes isolated fliers (the 13-55 mm-spacing blur points in your data)
    without touching the dense core. Returns filtered points.
    """
    if xyz.shape[0] == 0:
        return xyz
    tree = cKDTree(xyz)
    counts = tree.query_ball_point(xyz, r=radius, return_length=True)
    keep = counts > min_neighbors          # >min because it counts self
    return xyz[keep]


def statistical_outlier_filter(xyz: np.ndarray, k: int = 16,
                               std_ratio: float = 2.0) -> np.ndarray:
    """Drop points whose mean distance to k-NN is a cloud-level outlier."""
    if xyz.shape[0] <= k + 1:
        return xyz
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)
    md = d[:, 1:].mean(axis=1)
    keep = md < md.mean() + std_ratio * md.std()
    return xyz[keep]


def voxel_resample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """One averaged point per voxel -> uniform density, shape preserved."""
    if xyz.shape[0] == 0 or voxel <= 0:
        return xyz
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    sums = np.zeros((n, 3)); cnts = np.zeros(n)
    np.add.at(sums, inv, xyz); np.add.at(cnts, inv, 1.0)
    return sums / cnts[:, None]


def reconstruct_object(clouds: list[np.ndarray],
                       per_frame_radius: float = 0.01,
                       per_frame_min_neighbors: int = 4,
                       consistency_voxel: float = 0.005,
                       consistency_min_views_frac: float = 0.0,
                       final_voxel: float = 0.005,
                       sor_k: int = 16, sor_std: float = 2.0,
                       verbose: bool = True) -> np.ndarray:
    """
    Full sparse-cloud reconstruction. Returns the clean object cloud.

    Key params:
      per_frame_radius / min_neighbors : Stage A flier removal, per view.
      consistency_voxel                : cell size at which cross-view support
                                         is measured (Stage C).
      final_voxel                      : output density (Stage D).
    """
    def log(*a):
        if verbose: print(*a)

    # ---- A. per-frame filter: remove fliers BEFORE fusing ----
    filtered = []
    for c in clouds:
        f = radius_outlier_filter(c, per_frame_radius, per_frame_min_neighbors)
        if f.shape[0] > 0:
            filtered.append(f)
    log(f"A. per-frame filter: {sum(c.shape[0] for c in clouds)} -> "
        f"{sum(f.shape[0] for f in filtered)} pts across {len(filtered)} views")

    # ---- B. fuse: concatenate (shared world frame) ----
    fused = np.concatenate(filtered, axis=0)
    log(f"B. fused: {fused.shape[0]} pts")

    # ---- C. cross-view consistency: keep cells seen by enough support ----
    # A real surface cell accumulates many points (re-observed across views);
    # a noise cell has few. Threshold on per-cell point count.
    if consistency_min_views_frac > 0 and len(filtered) > 1:
        keys = np.floor(fused / consistency_voxel).astype(np.int64)
        uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                                      return_counts=True)
        # require a cell to hold at least this many points to survive
        min_count = max(2, int(consistency_min_views_frac * len(filtered)))
        good_cells = counts >= min_count
        keep = good_cells[inv]
        fused = fused[keep]
        log(f"C. consistency (>= {min_count} pts/cell): {fused.shape[0]} pts")
    else:
        log("C. consistency: skipped (single view or frac=0)")

    # ---- statistical outlier removal on the fused cloud ----
    before = fused.shape[0]
    fused = statistical_outlier_filter(fused, sor_k, sor_std)
    log(f"   SOR: {fused.shape[0]} pts (dropped {before - fused.shape[0]})")

    # ---- D. uniform resample ----
    out = voxel_resample(fused, final_voxel)
    log(f"D. resample @ {final_voxel*1000:.0f}mm: {out.shape[0]} pts (uniform)")
    return out


def oriented_box(xyz: np.ndarray):
    c = xyz.mean(0); X = xyz - c
    ev, evec = np.linalg.eigh(np.cov(X.T))
    R = evec[:, np.argsort(ev)[::-1]]
    proj = X @ R
    return c, R, (proj.max(0) - proj.min(0))
