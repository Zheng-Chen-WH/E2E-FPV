# 将四元数转成旋转矩阵
import numpy as np
def quaternions_to_rotation_matrices_np(quaternions):
    """
    将四元数转换为旋转矩阵。
    
    参数:
        quaternions: 一个形状为 (..., 4) 的NumPy数组，最后一个维度是 (w, x, y, z)。
        
    返回:
        一个形状为 (..., 3, 3) 的旋转矩阵NumPy数组。
    """
    # 归一化
    q_norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    quaternions = quaternions / q_norm
    
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    rot_mats = np.zeros(quaternions.shape[:-1] + (3, 3), dtype=np.float32)
    
    rot_mats[..., 0, 0] = 1 - 2 * (yy + zz)
    rot_mats[..., 0, 1] = 2 * (xy - wz)
    rot_mats[..., 0, 2] = 2 * (xz + wy)
    
    rot_mats[..., 1, 0] = 2 * (xy + wz)
    rot_mats[..., 1, 1] = 1 - 2 * (xx + zz)
    rot_mats[..., 1, 2] = 2 * (yz - wx)
    
    rot_mats[..., 2, 0] = 2 * (xz - wy)
    rot_mats[..., 2, 1] = 2 * (yz + wx)
    rot_mats[..., 2, 2] = 1 - 2 * (xx + yy)
    
    return rot_mats
dot_6 = quaternions_to_rotation_matrices_np(np.array([0.707, 0.707, 0, 0]))
print(dot_6)
print(np.stack([dot_6,dot_6,dot_6]).shape)