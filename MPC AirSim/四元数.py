import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义两个四元数
# q_a: 绕世界X轴旋转30度
q_a = R.from_euler('x', 30, degrees=True)
print(f"四元数 q_a (绕X轴30度) 的欧拉角 (XYZ顺序): {q_a.as_euler('xyz', degrees=True)}")
print(f"四元数 q_a 的分量 (x,y,z,w): {q_a.as_quat()}")

# q_b: 绕世界Y轴旋转60度
q_b = R.from_euler('y', 45, degrees=True)
print(f"\n四元数 q_b (绕Y轴45度) 的欧拉角 (XYZ顺序): {q_b.as_euler('xyz', degrees=True)}")
print(f"四元数 q_b 的分量 (x,y,z,w): {q_b.as_quat()}")

# --- 1. 计算 q_a * q_b ---
# 在Scipy中，R1 * R2 意味着 R1 作用在 R2 之后。
# 物理意义 (解释一): 先应用 q_b 的旋转，然后在此基础上应用 q_a 的旋转 (都在世界坐标系中)。
# 物理意义 (解释二): 假设 q_b 是当前姿态，q_a 是在 q_b 的本地坐标系中施加的额外旋转。
result_ab = q_a * q_b
print("\n--- 结果: q_a * q_b ---")
print(f"组合四元数 (q_a * q_b) 的欧拉角 (XYZ顺序): {result_ab.as_euler('xyz', degrees=True)}")
print(f"组合四元数 (q_a * q_b) 的分量 (x,y,z,w): {result_ab.as_quat()}")

# --- 2. 计算 q_b * q_a ---
# 物理意义 (解释一): 先应用 q_a 的旋转，然后在此基础上应用 q_b 的旋转 (都在世界坐标系中)。
# 物理意义 (解释二): 假设 q_a 是当前姿态，q_b 是在 q_a 的本地坐标系中施加的额外旋转。
result_ba = q_b * q_a
print("\n--- 结果: q_b * q_a ---")
print(f"组合四元数 (q_b * q_a) 的欧拉角 (XYZ顺序): {result_ba.as_euler('xyz', degrees=True)}")
print(f"组合四元数 (q_b * q_a) 的分量 (x,y,z,w): {result_ba.as_quat()}")

# --- 比较结果 ---
print("\n--- 比较 ---")
print(f"q_a * q_b == q_b * q_a? {np.allclose(result_ab.as_quat(), result_ba.as_quat())}")