"""
Utility functions for depth unprojection and 3D point transformations.

Pure numpy, no ROS dependencies.
"""

import numpy as np


def quaternion_to_rotation_matrix(qx: float, qy: float,
                                  qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3×3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    return np.array([
        [1 - 2 * (qy * qy + qz * qz),
         2 * (qx * qy - qw * qz),
         2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),
         1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),
         2 * (qy * qz + qw * qx),
         1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def unproject_patches_to_3d(
    depth_map: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    patch_size: int,
    input_size: int,
    orig_h: int, orig_w: int,
    n_patches_h: int, n_patches_w: int,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject DINOv2 patch centres to 3‑D points in the camera frame.

    For every patch (i, j) in the *resized* feature grid (input_size×input_size),
    we compute its centre pixel in the *original* image, look up the depth
    and back‑project using the pinhole model with the **original** camera
    intrinsics (fx, fy, cx, cy).

    Parameters
    ----------
    depth_map : (orig_h, orig_w) float32 – depth in metres (32FC1).
    fx, fy, cx, cy : camera intrinsics for the *original* resolution.
    patch_size : DINOv2 patch side in pixels (14 for ViT‑S/14).
    input_size : side of the square image fed to DINOv2 (518).
    orig_h, orig_w : original image dimensions (720, 1280).
    n_patches_h, n_patches_w : number of patches per axis (37, 37).
    depth_min, depth_max : valid depth range [m].

    Returns
    -------
    points_3d : (N, 3) float64 – 3‑D points in the camera frame
                (X right, Y down, Z forward).
    valid_mask : (N,) bool – True where depth was valid.
    """
    # Patch centres in resized (518×518) image coordinates.
    j_idx = np.arange(n_patches_w, dtype=np.float64)  # column index
    i_idx = np.arange(n_patches_h, dtype=np.float64)  # row index

    u_resized = j_idx * patch_size + patch_size / 2.0   # (n_patches_w,)
    v_resized = i_idx * patch_size + patch_size / 2.0   # (n_patches_h,)

    # Map back to original resolution.
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    u_orig = u_resized * scale_x   # (n_patches_w,)
    v_orig = v_resized * scale_y   # (n_patches_h,)

    # Full meshgrid – (n_patches_h, n_patches_w) each.
    uu, vv = np.meshgrid(u_orig, v_orig)

    # Integer pixel coordinates (clipped to image bounds).
    uu_int = np.clip(np.round(uu).astype(int), 0, orig_w - 1)
    vv_int = np.clip(np.round(vv).astype(int), 0, orig_h - 1)

    # Depth look‑up.
    Z = depth_map[vv_int, uu_int].astype(np.float64)

    # Validity mask.
    valid = np.isfinite(Z) & (Z > depth_min) & (Z < depth_max)

    # Pinhole back‑projection (camera frame: X right, Y down, Z forward).
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    points_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    valid_mask = valid.reshape(-1)

    return points_3d, valid_mask


def apply_transform(
    points: np.ndarray,
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transform to a set of 3‑D points.

    Parameters
    ----------
    points : (N, 3) – source‑frame points.
    translation : (3,) – translation vector (target frame origin in source).
    rotation_matrix : (3, 3) – rotation from source to target frame.

    Returns
    -------
    (N, 3) – points in the target frame.
    """
    return (rotation_matrix @ points.T).T + translation
