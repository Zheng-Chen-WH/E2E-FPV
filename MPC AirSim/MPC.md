**1.基于学习的 MPC (Learning-based MPC):**

学习动力学模型： 使用神经网络（如高斯过程、循环神经网络 RNNs、Transformers）从数据中学习系统动力学模型，特别是当精确的物理模型难以获得或过于复杂时。

学习成本函数/策略： 通过强化学习 (RL) 或模仿学习 (IL) 来学习 MPC 的成本函数参数，甚至直接学习控制策略的一部分。
残差学习： 学习名义模型与真实系统之间的误差（残差），并用于补偿。
安全学习： 结合学习方法和传统控制理论，提供安全性和性能保证。

**2.分布式 MPC (Distributed MPC):**

用于大规模多智能体系统（如机器人集群、电网、交通网络）。
每个智能体只基于局部信息和与邻居的通信来优化自己的控制策略，同时试图达成全局目标。
研究重点在于通信协议、收敛性、鲁棒性和可扩展性。

**3.随机 MPC (Stochastic MPC):**

明确考虑系统中的不确定性（过程噪声、测量噪声、参数不确定性）。
目标通常是最小化期望成本，或者满足概率约束（Chance-Constrained MPC）。
求解方法包括场景方法 (scenario approach)、多项式混沌展开 (polynomial chaos expansion) 等。MPPI 本身也可以看作一种处理不确定性的方法。

**4.鲁棒 MPC (Robust MPC):**

旨在保证在有界不确定性存在的情况下，系统性能（如稳定性、约束满足）仍然得到满足。
方法包括最小-最大优化 (min-max optimization)、使用不变集 (invariant sets) 等。

**5.自适应 MPC (Adaptive MPC):**

在线调整模型参数或控制器参数以适应系统时变特性或未建模动态。
常与系统辨识技术结合。

**6.经济 MPC (Economic MPC / EMPC):**

目标是优化更一般的经济性能指标，而不仅仅是跟踪设定点或轨迹。
成本函数可能不是正定的，可能与系统稳态运行相关。
研究重点在于保证稳定性的同时优化经济目标。

**7.显式 MPC (Explicit MPC):**

对于某些类别的系统（如线性系统和分段仿射成本函数），MPC 控制律可以预先离线计算出来，表示为状态空间的一个分段仿射函数。
优点是实时计算量极小，但离线计算复杂，且对状态空间维度敏感。

**8.快速 MPC 求解器 (Fast MPC Solvers):**

研究更高效的数值优化算法来求解 MPC 问题，例如基于内点法、主动集法、交替方向乘子法 (ADMM) 的改进。
利用问题结构（如稀疏性）进行加速。
硬件加速（FPGA, GPU）。

**MPC文件**

所有文件都是从0,0飞到20,10，始末速度都要求为0

1.MPC.py

最简单的MPC，SLSQP优化，随着horizon增长，计算耗时指数上升

2.MPPI.py

基于采样的MPC，生成大量轨迹，分别计算成本函数，加权平均得到控制序列

3.CEM-MPC.py

MPPI改进，由正态分布迭代生成少量轨迹，选择其中最优轨迹（求样本方差均值再作为下一次迭代取样分布），对最后一次迭代结果求均值作为序列

4.NN-CEM-MPC.py

针对AirSim动态模型黑箱，设计了神经网络拟合模型，demo中加入了控制指令随机生效系数，学习动态模型以正常应用MPC