"""
Real per-object 3D Gaussian Splatting, using gsplat's rasterizer AND its
DefaultStrategy for adaptive density control (clone / split / prune) -- the full
original-3DGS optimization, not a hand-rolled prune.

Structured like the sphere-visualizer it replaces (a class with __init__ +
reconstruct), but this is the actual method: it REQUIRES posed image crops and
runs an optimization loop. GS fits Gaussians to views by differentiable
rendering; it cannot be a one-shot point->mesh pass.

Why this is real Gaussian Splatting:
  - anisotropic Gaussians (means, quats, scales, opacities) -- soft oriented
    blobs, not hard spheres;
  - gsplat.rasterization -> differentiable splat render + alpha compositing;
  - gsplat.DefaultStrategy -> ADC densification (duplicate high-gradient small
    GSs, split large ones, prune transparent/oversized), managing optimizer
    state across the parameter changes;
  - novel-view render by splatting, not mesh rendering.

Requires on the GPU box (not needed just to import): torch(cu124) + gsplat.
Seed from the fused object point cloud you already produce.

Views come from what reconstruction_3d/object_tracker.py's Tracker._save
already writes per frame: <ts>_points.npy (world points), <ts>.png (this
frame's crop), and <ts>_pose.npz (R, t: that frame's camera-to-world pose;
bbox: the crop's [x_min,y_min,x_max,y_max] within the full frame, needed to
shift the principal point since K is defined for the full image, not the
crop). See view_reconstructed_object.py / view_objects_map.py for the loader
that turns those into View objects.

    gs = GaussianSplatter(iters=2000, cap_gaussians=150_000, device="cuda")
    gs.reconstruct(points, colors, views)
    img = gs.render(views[0])
    gs.export_ply("object.ply"); gs.save_model("object.pt")
"""
from __future__ import annotations
from dataclasses import dataclass
import glob
import os

import cv2
import numpy as np


@dataclass
class View:
    """One posed image crop of the object -- the supervision signal for GS."""
    T_world_cam: np.ndarray        # (4,4) camera pose, SAME frame as points
    K: np.ndarray                  # (3,3) intrinsics for this crop
    image: np.ndarray              # (H,W,3) float32 in [0,1]


def load_views(folder: str, fx: float, fy: float, cx: float, cy: float) -> list[View]:
    """Builds posed Views from what Tracker._save already wrote in one
    object folder: <ts>.png (this frame's crop) + <ts>_pose.npz (R, t: that
    frame's camera-to-world pose; bbox: the crop's location in the full
    frame). fx/fy/cx/cy are the full-image intrinsics (config/params.yaml);
    K's principal point is shifted by the crop's [x_min, y_min] since the
    crop is a sub-window of the full image, not the full image itself.
    """
    views = []
    for pose_path in sorted(glob.glob(os.path.join(folder, '*_pose.npz'))):
        stamp = os.path.basename(pose_path)[:-len('_pose.npz')]
        crop_path = os.path.join(folder, f'{stamp}.png')
        if not os.path.isfile(crop_path):
            continue

        data = np.load(pose_path)
        R, t, bbox = data['R'], data['t'], data['bbox']
        x_min, y_min = bbox[0], bbox[1]

        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R
        T_world_cam[:3, 3] = t

        K = np.array([[fx, 0, cx - x_min],
                      [0, fy, cy - y_min],
                      [0, 0, 1]], dtype=np.float32)

        crop_bgr = cv2.imread(crop_path)
        image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        views.append(View(T_world_cam=T_world_cam, K=K, image=image))
    return views


def _knn_mean_dist(points: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(points).query(points, k=k + 1)
    return d[:, 1:].mean(axis=1)


class GaussianSplatter:

    def __init__(self, iters: int = 2000, cap_gaussians: int = 150_000,
                 sh_degree: int = 0,
                 lr_means: float = 1.6e-4, lr_scales: float = 5e-3,
                 lr_quats: float = 1e-3, lr_colors: float = 2.5e-3,
                 lr_opacities: float = 5e-2,
                 refine_start: int = 300, refine_stop: int | None = None,
                 refine_every: int = 100, reset_every: int = 3000,
                 device: str = "cuda"):
        self.iters = iters
        self.cap = cap_gaussians
        self.sh_degree = sh_degree          # 0 = plain RGB (cheapest, 4GB-friendly)
        self.lrs = dict(means=lr_means, scales=lr_scales, quats=lr_quats,
                        colors=lr_colors, opacities=lr_opacities)
        self.refine_start = refine_start
        self.refine_stop = refine_stop if refine_stop is not None else int(0.75 * iters)
        self.refine_every = refine_every
        self.reset_every = reset_every
        self.device = device
        self.params = None                  # torch.nn.ParameterDict
        self.optimizers = None              # per-parameter Adam dict

    # ---------- init Gaussians from the fused cloud ----------
    def _init(self, points: np.ndarray, colors: np.ndarray):
        import torch
        scale = _knn_mean_dist(points, k=3)
        scales = np.log(np.clip(np.repeat(scale[:, None], 3, axis=1), 1e-6, None))
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (points.shape[0], 1))   # wxyz identity
        opac = np.full(points.shape[0], np.log(0.1 / 0.9))            # logit(0.1)
        dev = self.device

        def P(a):
            return torch.nn.Parameter(torch.tensor(a, dtype=torch.float32, device=dev))

        # DefaultStrategy requires keys {"means","scales","quats","opacities"};
        # we carry "colors" too (rasterized as RGB when sh_degree=0).
        self.params = torch.nn.ParameterDict({
            "means":     P(points),
            "scales":    P(scales),
            "quats":     P(quats),
            "opacities": P(opac),
            "colors":    P(np.clip(colors, 1e-4, 1 - 1e-4)),
        })
        # per-parameter optimizers -- the layout gsplat strategies expect so they
        # can add/remove rows from Adam's state when Gaussians are cloned/pruned.
        self.optimizers = {
            k: torch.optim.Adam([self.params[k]], lr=self.lrs[k])
            for k in self.params.keys()
        }

    def _activated(self):
        import torch
        p = self.params
        return dict(
            means=p["means"],
            scales=torch.exp(p["scales"]),
            quats=torch.nn.functional.normalize(p["quats"], dim=-1),
            opacities=torch.sigmoid(p["opacities"]),
            colors=torch.clamp(p["colors"], 0.0, 1.0),
        )

    def _render(self, view: "View"):
        import torch
        from gsplat import rasterization
        g = self._activated()
        H, W = view.image.shape[:2]
        viewmat = torch.linalg.inv(
            torch.tensor(view.T_world_cam, dtype=torch.float32, device=self.device))
        K = torch.tensor(view.K, dtype=torch.float32, device=self.device)
        renders, alphas, info = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"],
            opacities=g["opacities"], colors=g["colors"],
            viewmats=viewmat[None], Ks=K[None], width=W, height=H,
            packed=True, render_mode="RGB",
        )
        return renders[0], alphas[0], info      # (H,W,3), (H,W,1), meta dict

    # ---------- the fit loop with DefaultStrategy densification ----------
    def reconstruct(self, points: np.ndarray, colors: np.ndarray | None = None,
                    views: list["View"] = None, verbose: bool = True):
        import torch
        from gsplat import DefaultStrategy
        if not views:
            raise ValueError("GS needs posed image crops (views) to fit against. "
                             "Points alone -> use the sphere visualizer instead.")
        if colors is None:
            colors = np.full((points.shape[0], 3), 0.5, np.float32)
        self._init(points, colors)

        # NOTE: this gsplat version's DefaultStrategy has no hard Gaussian
        # count cap (that's MCMCStrategy's cap_max, a different strategy) --
        # growth here is governed purely by the gradient/scale thresholds
        # below, so self.cap is currently advisory only, not enforced.
        strategy = DefaultStrategy(
            prune_opa=0.005, grow_grad2d=2e-4, grow_scale3d=0.01,
            refine_start_iter=self.refine_start,
            refine_stop_iter=self.refine_stop,
            refine_every=self.refine_every, reset_every=self.reset_every,
            verbose=False,
        )
        strategy.check_sanity(self.params, self.optimizers)
        state = strategy.initialize_state()

        # Ground truth stays on the CPU -- only the one view sampled per step
        # is moved to the GPU. Holding all of them resident costs
        # len(views) * H*W*3 * 4 bytes of VRAM for no benefit (453 MB for 123
        # 640x480 views), which matters on a small card.
        gt = [torch.tensor(v.image, dtype=torch.float32) for v in views]

        for step in range(self.iters):
            i = np.random.randint(len(views))
            render, _, info = self._render(views[i])

            # strategy needs the 2D-means gradient retained for its ADC decisions
            strategy.step_pre_backward(self.params, self.optimizers, state, step, info)
            loss = torch.abs(render - gt[i].to(self.device)).mean()   # L1 photometric
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in self.optimizers.values():
                opt.step()
            # clone / split / prune (updates params + optimizers in place).
            # packed=True matches _render()'s rasterization(..., packed=True)
            # call above -- info's layout depends on it, and the default
            # (False) would silently misread a packed-mode info dict.
            strategy.step_post_backward(self.params, self.optimizers, state,
                                        step, info, packed=True)

            if verbose and step % max(1, self.iters // 10) == 0:
                print(f"  step {step:4d}/{self.iters}  L1 {loss.item():.4f}  "
                      f"gaussians {self.params['means'].shape[0]}")
        return self

    # ---------- outputs ----------
    def render(self, view: "View") -> np.ndarray:
        import torch
        with torch.no_grad():
            img, _, _ = self._render(view)
            return (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def export_ply(self, path: str):
        """Gaussian centers + colors as a colored PLY (viewable anywhere)."""
        import torch
        with torch.no_grad():
            g = self._activated()
            xyz = g["means"].cpu().numpy()
            rgb = (g["colors"].cpu().numpy() * 255).astype(np.uint8)
        n = xyz.shape[0]
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    def save_model(self, path: str):
        """Full Gaussian params (means, scales, quats, opacities, colors)."""
        import torch
        torch.save({k: v.detach().cpu() for k, v in self.params.items()}, path)
