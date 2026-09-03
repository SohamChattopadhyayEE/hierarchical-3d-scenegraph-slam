"""Poisson surface reconstruction: turns a clean sparse point cloud (e.g.
reconstruct_object()'s output) into a dense, smooth mesh for visualization.

open3d does the actual Poisson solve -- reimplementing that by hand is out
of scope, same reasoning as using ultralytics/mobile_sam for segmentation
instead of hand-rolling their decode logic.
"""

import numpy as np
import open3d as o3d


class PoissonMesher:

    def __init__(self, depth: int = 8, density_trim_quantile: float = 0.05):
        self.depth = depth
        self.density_trim_quantile = density_trim_quantile

    def reconstruct(self, points: np.ndarray, colors: np.ndarray = None):
        """points: Nx3 world-frame points. colors: optional Nx3 in [0,1].
        Returns an open3d.geometry.TriangleMesh."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=10)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth)

        # Poisson extrapolates a closed surface everywhere, including
        # regions with no real point support -- trim those low-density
        # (hallucinated) vertices so only the actually-observed shape remains.
        densities = np.asarray(densities)
        threshold = np.quantile(densities, self.density_trim_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

        return mesh
