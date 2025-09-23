"""
Mesh utilities for object loading and SDF computation.

Handles loading BEHAVE object meshes, point sampling, and computing
signed distance fields for interaction geometry (ig) vectors.

Key features:
- Load and normalize object meshes to zero-mean canonical frame
- Pre-sample object points for efficient SDF computation
- Compute point-to-mesh distances for contact detection
- Transform points to world coordinates via object pose
- Generate heading-aligned interaction geometry vectors

References:
- InterMimic: SDF-based interaction geometry vectors (InterMimic paper)
- BEHAVE: Object mesh data and canonical frames (BEHAVE paper)
"""

import torch
import numpy as np
import trimesh
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

from .quat_utils import ensure_tensor, rotate_points_by_quat, calc_heading_quat_inv

logger = logging.getLogger(__name__)

class ObjectMeshProcessor:
    """
    Processor for BEHAVE object meshes with SDF computation capabilities.

    This class handles loading object meshes, normalizing to canonical frame,
    and computing interaction geometry vectors for InterMimic training.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize mesh processor.

        Args:
            device: Device for tensor computation
        """
        self.device = device
        self.mesh_cache = {}  # Cache loaded meshes
        self.points_cache = {}  # Cache sampled points

    def load_object_mesh(self, mesh_path: Union[str, Path]) -> trimesh.Trimesh:
        """
        Load object mesh from file.

        Args:
            mesh_path: Path to object mesh (.obj file)

        Returns:
            mesh: Loaded mesh

        Raises:
            FileNotFoundError: If mesh file doesn't exist
            ValueError: If mesh loading fails
        """
        mesh_path = Path(mesh_path)

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        if mesh_path in self.mesh_cache:
            return self.mesh_cache[mesh_path]

        try:
            mesh = trimesh.load(mesh_path, force='mesh')

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Failed to load as trimesh: {mesh_path}")

            # Validate mesh
            if mesh.vertices.shape[0] == 0:
                raise ValueError(f"Mesh has no vertices: {mesh_path}")

            if mesh.faces.shape[0] == 0:
                raise ValueError(f"Mesh has no faces: {mesh_path}")

            # Cache mesh
            self.mesh_cache[mesh_path] = mesh

            logger.info(f"Loaded mesh: {mesh_path} "
                       f"({mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces)")

            return mesh

        except Exception as e:
            raise ValueError(f"Failed to load mesh {mesh_path}: {e}")

    def normalize_mesh_to_canonical(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize mesh to zero-mean canonical frame.

        This centers the mesh at origin for consistent point sampling
        and SDF computation. The original mesh file is not modified.

        Args:
            mesh: Input mesh

        Returns:
            normalized_mesh: Mesh with vertices centered at origin
        """
        # Copy mesh to avoid modifying original
        normalized_mesh = mesh.copy()

        # Center vertices at origin
        centroid = normalized_mesh.vertices.mean(axis=0)
        normalized_mesh.vertices -= centroid

        logger.info(f"Normalized mesh to canonical frame (centroid: {centroid})")

        return normalized_mesh

    def sample_object_points(self, mesh: trimesh.Trimesh, n_points: int = 2048) -> torch.Tensor:
        """
        Sample points from object mesh surface.

        Args:
            mesh: Mesh to sample from
            n_points: Number of points to sample

        Returns:
            points: (n_points, 3) sampled points on mesh surface
        """
        cache_key = (id(mesh), n_points)

        if cache_key in self.points_cache:
            return self.points_cache[cache_key]

        try:
            # Sample points uniformly on mesh surface
            points, _ = trimesh.sample.sample_surface(mesh, n_points)

            # Convert to tensor
            points_tensor = torch.from_numpy(points).float().to(self.device)

            # Cache points
            self.points_cache[cache_key] = points_tensor

            logger.info(f"Sampled {n_points} points from mesh surface")

            return points_tensor

        except Exception as e:
            raise ValueError(f"Failed to sample points from mesh: {e}")

    def transform_points_to_world(self,
                                canonical_points: torch.Tensor,
                                obj_pos: torch.Tensor,
                                obj_rot: torch.Tensor) -> torch.Tensor:
        """
        Transform canonical object points to world coordinates.

        Args:
            canonical_points: (N, 3) points in canonical frame
            obj_pos: (T, 3) object positions
            obj_rot: (T, 4) object rotations [x,y,z,w]

        Returns:
            world_points: (T, N, 3) points in world coordinates
        """
        canonical_points = ensure_tensor(canonical_points, self.device)
        obj_pos = ensure_tensor(obj_pos, self.device)
        obj_rot = ensure_tensor(obj_rot, self.device)

        T = obj_pos.shape[0]
        N = canonical_points.shape[0]

        # Expand canonical points to match temporal dimension
        points_expanded = canonical_points.unsqueeze(0).expand(T, N, 3)

        # Rotate points by object rotation
        rotated_points = rotate_points_by_quat(points_expanded, obj_rot.unsqueeze(1))

        # Translate by object position
        world_points = rotated_points + obj_pos.unsqueeze(1)

        return world_points

    def compute_point_to_mesh_distances(self,
                                      query_points: torch.Tensor,
                                      mesh_points: torch.Tensor,
                                      return_vectors: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute distances from query points to mesh surface.

        This provides a fast approximation of SDF using nearest neighbor
        distances to pre-sampled mesh points.

        Args:
            query_points: (T, M, 3) query points (e.g., body joints)
            mesh_points: (T, N, 3) mesh surface points
            return_vectors: If True, return distance vectors, else distances only

        Returns:
            distances: (T, M) distances to mesh surface
            vectors: (T, M, 3) vectors from query to nearest mesh points (if return_vectors)
        """
        query_points = ensure_tensor(query_points, self.device)
        mesh_points = ensure_tensor(mesh_points, self.device)

        T, M = query_points.shape[:2]
        N = mesh_points.shape[1]

        # Compute pairwise distances
        # query_points: (T, M, 3) -> (T, M, 1, 3)
        # mesh_points: (T, N, 3) -> (T, 1, N, 3)
        query_expanded = query_points.unsqueeze(2)  # (T, M, 1, 3)
        mesh_expanded = mesh_points.unsqueeze(1)    # (T, 1, N, 3)

        # Distance vectors: (T, M, N, 3)
        diff_vectors = query_expanded - mesh_expanded

        # Distances: (T, M, N)
        distances_pairwise = torch.norm(diff_vectors, dim=-1)

        # Find nearest mesh point for each query point
        min_distances, min_indices = torch.min(distances_pairwise, dim=-1)  # (T, M)

        if not return_vectors:
            return min_distances

        # Get vectors to nearest points
        # min_indices: (T, M) -> (T, M, 1)
        min_indices_expanded = min_indices.unsqueeze(-1)

        # Select corresponding vectors: (T, M, 3)
        batch_indices = torch.arange(T, device=self.device).unsqueeze(1).expand(T, M)
        query_indices = torch.arange(M, device=self.device).unsqueeze(0).expand(T, M)

        nearest_vectors = diff_vectors[batch_indices, query_indices, min_indices]

        return min_distances, nearest_vectors

    def compute_contact_labels(self,
                             distances: torch.Tensor,
                             threshold: float = 0.02) -> torch.Tensor:
        """
        Compute binary contact labels from distances.

        Args:
            distances: (T, M) distances to object surface
            threshold: Contact threshold in meters (default: 2cm)

        Returns:
            contacts: (T, M) binary contact labels
        """
        distances = ensure_tensor(distances, self.device)
        contacts = (distances < threshold).float()

        return contacts

    def compute_interaction_geometry(self,
                                   body_pos: torch.Tensor,
                                   obj_pos: torch.Tensor,
                                   obj_rot: torch.Tensor,
                                   root_rot: torch.Tensor,
                                   canonical_points: torch.Tensor) -> torch.Tensor:
        """
        Compute interaction geometry (ig) vectors for InterMimic.

        This follows InterMimic's approach of:
        1. Transform object points to world coordinates
        2. Compute distance vectors from body joints to object surface
        3. Transform vectors to heading-aligned local frame
        4. Flatten to 1D feature vector

        Args:
            body_pos: (T, 52, 3) body joint positions
            obj_pos: (T, 3) object positions
            obj_rot: (T, 4) object rotations [x,y,z,w]
            root_rot: (T, 4) root rotations for heading alignment [x,y,z,w]
            canonical_points: (N, 3) canonical object points

        Returns:
            ig: (T, 52*3) interaction geometry vectors
        """
        body_pos = ensure_tensor(body_pos, self.device)
        obj_pos = ensure_tensor(obj_pos, self.device)
        obj_rot = ensure_tensor(obj_rot, self.device)
        root_rot = ensure_tensor(root_rot, self.device)
        canonical_points = ensure_tensor(canonical_points, self.device)

        T = body_pos.shape[0]

        # 1. Transform object points to world coordinates
        world_points = self.transform_points_to_world(canonical_points, obj_pos, obj_rot)

        # 2. Compute distance vectors from body joints to object surface
        distances, vectors = self.compute_point_to_mesh_distances(
            body_pos, world_points, return_vectors=True
        )

        # 3. Transform vectors to heading-aligned local frame
        heading_rot_inv = calc_heading_quat_inv(root_rot)

        # Apply heading rotation to each vector
        # vectors: (T, 52, 3), heading_rot_inv: (T, 4)
        local_vectors = rotate_points_by_quat(vectors, heading_rot_inv.unsqueeze(1))

        # 4. Flatten to 1D feature vector
        ig = local_vectors.reshape(T, -1)  # (T, 52*3)

        return ig

    def get_object_canonical_points(self,
                                  object_mesh_root: Union[str, Path],
                                  object_category: str,
                                  n_points: int = 2048) -> torch.Tensor:
        """
        Get canonical points for object category.

        Args:
            object_mesh_root: Root directory for object meshes
            object_category: Object category name
            n_points: Number of points to sample

        Returns:
            canonical_points: (n_points, 3) canonical object points

        Raises:
            FileNotFoundError: If object mesh file not found
        """
        mesh_path = Path(object_mesh_root) / object_category / f"{object_category}.obj"

        # Load mesh
        mesh = self.load_object_mesh(mesh_path)

        # Normalize to canonical frame
        canonical_mesh = self.normalize_mesh_to_canonical(mesh)

        # Sample points
        canonical_points = self.sample_object_points(canonical_mesh, n_points)

        return canonical_points

def approximate_sdf_batch(query_points: torch.Tensor,
                         mesh_points: torch.Tensor,
                         inside_threshold: float = 0.01) -> torch.Tensor:
    """
    Approximate signed distance field using point cloud.

    This is a fast approximation that assigns negative distances to points
    that are very close to the mesh (assumed inside) and positive otherwise.

    Args:
        query_points: (T, M, 3) query points
        mesh_points: (T, N, 3) mesh surface points
        inside_threshold: Threshold for considering points "inside"

    Returns:
        sdf: (T, M) approximated signed distances
    """
    # Get unsigned distances
    distances = ObjectMeshProcessor().compute_point_to_mesh_distances(
        query_points, mesh_points, return_vectors=False
    )

    # Simple heuristic: very close points are considered "inside"
    sdf = torch.where(distances < inside_threshold, -distances, distances)

    return sdf

def compute_object_contact_any(contact_human: torch.Tensor) -> torch.Tensor:
    """
    Compute object contact flag from human joint contacts.

    Args:
        contact_human: (T, 52) binary contact flags for human joints

    Returns:
        contact_obj: (T, 1) binary flag indicating any human-object contact
    """
    contact_human = ensure_tensor(contact_human)

    # Any human joint in contact -> object contact
    contact_obj = (contact_human.sum(dim=-1, keepdim=True) > 0).float()

    return contact_obj

def validate_mesh_file(mesh_path: Union[str, Path]) -> bool:
    """
    Validate that mesh file exists and is loadable.

    Args:
        mesh_path: Path to mesh file

    Returns:
        is_valid: True if mesh is valid
    """
    try:
        mesh_path = Path(mesh_path)

        if not mesh_path.exists():
            return False

        if not mesh_path.suffix.lower() in ['.obj', '.ply', '.stl']:
            return False

        # Try loading
        mesh = trimesh.load(mesh_path, force='mesh')

        if not isinstance(mesh, trimesh.Trimesh):
            return False

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return False

        return True

    except Exception:
        return False