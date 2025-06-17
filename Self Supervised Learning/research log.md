# 工作日志
**已经进行的改进：** 

1. 采用MPC作为专家监督学习，对端到端网络进行预训练（5.12）  
2. MPC、MPPI、CEM-MPPI经过对比实验，发现CEM方法在预测步长N增大时计算效率最高，决定采用CEM-MPPI作为基准MPC
3. 考虑到AirSim动力学环境几乎黑箱，采用神经网络拟合AirSim动力学特性，采用simpleflight作为底层控制器，MPC输出速度指令，神经网络输入状态（位置、速度、角速度）+速度指令，输出预测下一时刻状态
4. NN+MPPI计算速度太慢，采用GPU并行化采样，大大加快了计算速度（0.005s， 采样数100， N=15）；学习了爱因斯坦求和einsum
5. 输入状态增加了姿态角
6. MPC加入了动态目标追踪，即在预测序列内目标位置、速度同样随时间变化
7. MPC改变学习目标，不再以门中心为目标，而是以门中心前0.5m一点为目标，同时加入y方向速度目标，试图依托惯性实现冲门
8. 考虑到y方向位置、速度状态值过大，可能淹没其他特征，神经网络加入动态归一化，使用buffer数据计算均值与方差。buffer中存储原始数据，经过归一化后输出给神经网络预测状态，由于线性缩放后代价函数排序不变，不影响MPPI
9. 代码过于臃肿，考虑到后续改进便利性，进行了代码拆分重构，现在包括五个尽量模块化的文件
10. 通过疯狂print，确认了问题出在神经网络预测功能；其对X方向动力学预测很混乱 （5.19）
## 5.20
+ 通过打印buffer中的数据，确认了步长DT=0.1s时，airsim无法正确响应，可能是导致神经网络学习混乱的原因；DT=0.1s时得到的模型和buffer存在了DT=0.1s文件夹可能的解决方案：  
  + 增大DT，使无人机可以跟踪速度指令；尝试0.5sDT，N相应调整
    + 发现N=5时训出来的网络到N=10就用不了了（一直坠机），从N=10从头训一下
    + N=10从头训也不行，但是N=5，DT=0.5的时候训出来的成功率还算高，存在N=5，DT=0.5下面
  + 转移到PX4，直接输出更底层的加速度指令
+ 发现潜在创新点：UZH论文中无人机所有穿门时刻都是门在最高点（速度最慢处），无人机实际轨迹是跟踪门然后在最高点穿过去；如果实现在门快速运动的时刻也能穿越？

## 5.21
折腾了一天PX4转移，一直在试图解决拒绝加电的问题；本来以为PX4是个飞控程序，结果怎么是传感+飞控，甚至还有电池问题...

## 5.22
+ 继续折腾disarming，这次试试外部信息注入
+ gemini给出的控制使用外部信息的参数名称几乎都有问题，在QGC里找不到
  + 偶然翻到了正确参数：EKF2_AGP_CTRL, EKF2_BARO_CTRL, EKF2_EV_CTRL (全部勾选以开启外部信息输入), EKF2_GPS_CTRL, EKF2_DRAG_CTRL, EKF2_HGT_REF, EKF2_MAG_TYPE, 等，关键词搜"EKF2"或"CTRL"
  + 改了太多参数导致自检都完成不了 愤怒 全部reset之后记录修改参数一个一个排除法试
    + EKF2_HGT_REF：从GPS改成气压计再改成version，暂时没问题
    + 关掉EKF2_AGP_CTRL：0
    + 关掉EKF2_GPS_CTRL:0 无人机无法正常ready for takeoff，但是提供信息注入之后可以了；还是会arming denied，打印出来的状态也都是错的
    + 怎么现在airsim传出的位置数据都是错的了啊
    + 发现PX4 v1.11才有EKF2_AID_MASK，PX4 1.15直接把所有的传感器启用单独拿出来给了个参数

## 5.23
+ 最后再挣扎一下把PX4降到v1.11试试
  + 编译错误，/media/zheng/A214861F1485F697/PX4/PX4-Autopilot/platforms/common/px4_work_queue/WorkQueueManager.cpp的第257行从const size_t stacksize_adj = math::max(PTHREAD_STACK_MIN, PX4_STACK_ADJUSTED(wq->stacksize));改成了const size_t stacksize_adj = math::max((size_t)PTHREAD_STACK_MIN, PX4_STACK_ADJUSTED(wq->stacksize));
  + 还是不行，再改成const size_t stacksize_adj = math::max((size_t)PTHREAD_STACK_MIN, (size_t)PX4_STACK_ADJUSTED(wq->stacksize));
+ **放弃了 回去接着折腾airsim的更底层控制了 气死了**
+ 研究AirSim默认无人机物理参数和Simpleflight控制，搞一个简易动力学模型，然后只让神经网络学残差→感觉会难度低一些；可能有用的参数配置文件：
  + /media/zheng/A214861F1485F697/Airsim/AirLib/include/vehicles/multirotor/**MultiRotorParams.hpp**：定义了多旋翼无人机所需的所有核心物理参数，以及一些常用机型的默认配置；
  + 同上文件夹/**MultiRotorPhysicsBody.hpp**：PhysicsBody 基类的派生类，专门为多旋翼无人机实现了其独特的物理行为，揭示了 SimpleFlight 物理引擎如何将settings.json 中配置的参数（或默认参数）以及控制指令，转化为无人机的实际运动。
    + 它：加载和使用 MultiRotorParams 中定义的无人机物理参数。
    + 创建并管理每个旋翼 (RotorActuator)，将您的控制指令传递给它们。
    + 创建并管理空气阻力模型 (drag_faces_)。
    + 将这些力（来自旋翼和阻力）汇总起来，传递给基类 PhysicsBody，由 PhysicsBody 使用牛顿-欧拉方程计算出无人机的实际运动（位置、姿态、速度、加速度）。
    + 更新传感器和与外部API交互。
  + 同上文件夹/**RotorParams.hpp**:将控制指令最终转化为无人机物理推力和扭矩, 包含了单个旋翼的物理特性，以及它们如何影响整个无人机的动力学
    + RotorTurningDirection 枚举：RotorTurningDirectionCCW = -1, RotorTurningDirectionCW = 1。
      + 这定义了旋翼的旋转方向。在 MultiRotorPhysicsBody 中，这个方向会影响旋翼产生的反扭矩（即偏航力矩）的方向。例如，对于一个四旋翼，对角的两个旋翼通常旋转方向相同，而相邻的则相反，以抵消大部分反扭矩，只留下可以控制的偏航力矩。
    + struct RotorParams - 单个旋翼的物理参数和模型
  + 同上文件夹/**RotorActuator.hpp**:描述了单个旋翼（电机+螺旋桨）如何工作，以及它如何根据输入的控制信号产生物理力
    + RotorActuator 继承自 PhysicsBodyVertex：这证实了每个旋翼都被视为一个“物理作用点”，它能够向 PhysicsBody 施加力和力矩。
    + Output 结构体：定义了旋翼的输出：推力、扭矩大小、转速、旋转方向以及原始和滤波后的控制信号。
    + 构造函数和 initialize 方法：
      + position, normal, turning_direction: 这些来自 MultiRotorParams，定义了旋翼的几何位置和方向。
      + const RotorParams& params: 整个 RotorParams 对象被传递进来，并存储为 params_ 成员。这意味着 RotorActuator 会使用 RotorParams 中定义的 C_T, C_P, max_rpm 等参数来计算推力和扭矩。
      + control_signal_filter_.initialize(params_.control_signal_filter_tc, 0, 0);: 这里初始化了一个一阶低通滤波器，用于平滑控制信号。params_.control_signal_filter_tc 就是在 RotorParams 中定义的 control_signal_filter_tc。
      + static void setOutput(...) 方法：核心的数学模型
        + output.control_signal_filtered: 这是经过低通滤波器平滑后的0-1的控制信号。
        + output.speed = sqrt(output.control_signal_filtered * params.max_speed_square);: 定义了控制信号与旋翼转速（角速度）之间的关系。它假设旋翼的转速平方与控制信号（或油门输入）成正比。这符合许多电机模型的特性，即推力与转速平方成正比。
        + params.max_speed_square 是 (max_rpm / 60 * 2 * PI)^2。因此，output.speed 是当前控制信号对应的转速（弧度/秒）。
        + output.thrust = output.control_signal_filtered * params.max_thrust;: 这行也至关重要！ 它定义了推力与控制信号的关系。这意味着推力与滤波后的控制信号呈线性关系。结合上一行，这表示 SimpleFlight 实际上是在假设：推力与转速的平方成正比，并且转速的平方与控制信号成正比。 （即 Thrust ~ speed^2 ~ control_signal）。
        + output.torque_scaler = output.control_signal_filtered * params.max_torque * static_cast<int>(turning_direction);: 反扭矩也与滤波后的控制信号呈线性关系，并由 turning_direction 赋予方向。
    + 与控制器设计相关：
      + 明确了控制信号到推力的映射：SimpleFlight 中，您在Python API中设置的0-1的电机PWM值，经过低通滤波后，直接线性地映射到了推力（output.control_signal_filtered * params.max_thrust）。
      + 转速模型： output.speed = sqrt(output.control_signal_filtered * params.max_speed_square); 表明转速与控制信号的平方根成正比。
      + 推力与转速的关系： 由于 output.thrust = output.control_signal_filtered * params.max_thrust 且 output.control_signal_filtered = (output.speed)^2 / params.max_speed_square，因此，output.thrust = (output.speed)^2 / params.max_speed_square * params.max_thrust。这相当于 Thrust = K * speed^2 的形式，其中 K = params.max_thrust / params.max_speed_square。这与经典的螺旋桨推力公式 F = C_T * rho * n^2 * D^4 是吻合的。
      + control_signal_filter_tc： 调整这个时间常数可以模拟真实电机响应的延迟。如果您希望响应更快，可以减小这个值（但过小可能导致不稳定或不真实）。
      + 控制器设计：当外部控制器计算出所需的总推力 F_total 和总力矩 \tau_total 后，您需要将其分解到每个旋翼的推力 F_i 和反扭矩 \tau_i。

        然后，对于每个旋翼，您需要反向计算所需的控制信号： control_signal_i = F_i / params.max_thrust，将这些 control_signal_i 值（0-1之间）通过 moveByMotorPWMsAsync 发送给AirSim。

**simpleflight控制**：SimpleFlight 的核心思想是将控制指令转化为旋翼的推力与反扭矩，再将这些力矩和空气阻力等汇总，最终通过牛顿-欧拉方程更新无人机的运动状态。具体过程参见controller development.md

**gemini提议的settings.json修改方式**：添加或修改 RotorParams 子部分："Vehicles": { "SimpleFlight": { "Params": { "RotorParams": { ... } } } }

+ 把airsim里的无人机ue4模型缩小到了50%（0.5*0.5m）的大小，在ue4中启动场景（否则没有无人机）然后/AirSim/Blueprints/BP_FlyingPawn 选择 components tab -> Transform -> scale；由于模型是生成的，所以要点进mesh界面更改缩放才能每次生成都是缩放后的大小

## 5.26 5.27
+ 完成了动力学模型的开发，明天测试一下

## 5.28
+ 模型存在较大误差，进行修正
  + 角度方面误差：把旋转阻力系数改成0之后解决；否则会直接稳定在力矩=阻力矩的位置
  + 线速度与位置方面误差：引入基于drag_face的阻力计算方法，具体为：
    + 将阻力盒建模为一个正方体，x,y,z方向大小分别为机身盒与桨叶尺寸在相应方向的面积之和
    + 根据姿态将阻力盒投影到世界坐标系下，利用公式 $F=C_d \cdot V^2$ 计算阻力，其中 $Cd=0.5 \cdot 0.325\cdot A$,A为对应方向面积,0.325为文档给出的阻力系数
  + 还是有问题，x方向角速度误差很大，y方向角速度几乎没有误差，为什么会这样？
    + 没事了，x方向力矩表达式把左右电机正负号搞反了

## 5.29
+ 模型精度已经非常高了，不需要网络预测残差了

## 6.4
+ 加入并行化的代码整合好了，但是无人机行为还是很诡异，怀疑两个方面：
  1. 计算耗时太高；
  2. 并行化之后的模型精度出问题了
+ 用了实例化另一个模型的方法检测了一下精度，看起来没啥问题...
+ 怀疑是MPC的问题，调参试试，gemini建议先加个姿态目标约束
+ 靠着权重矩阵调参实现了穿门；在穿门之后换用另一套权重矩阵，即分段不同权重矩阵
+ 油门0.63-0.66就够快速穿门了
+ 迭代1次也可以，精英群体调到0.1;只要权重调好就不用担心这个比例太小

## 6.11
+ 完成了MPC执行监督学习代码的编写
+ 门的时序控制在加入拍照暂停环节之后不再合适了，转为采用elapsed_time控制的代码时间轴
+ 穿不了第二个门，对第二个门设计了强调x位置、z位置，弱化y速度的代价函数
+ 训练发现loss很差，不能收敛
  + 把policy的优化器换成AdamW，weight_decay先设成0.01
  + 学习率加入了warm_up机制（逐渐从0升高）
  + 基础学习率降到5e-5
  + 全连接网络增加到四层

## 6.12
+ 改成每step都训练，效果好多了
+ NN直接输出范围从-1~1改成-5~5

## 6.13
+ 测试保存结果，每个模型测试20次计算平均奖励

|模型文件名|平均奖励|
|-------------------------------|----------|
|loss_39138_0.008137084543704987|-1126.7585|
|loss_26021_0.008864268660545349|-1117.2075|
|loss_24000_0.010233676061034203|-648.1272|
|loss_23341_0.011256770230829716|263.7337|
|loss_21213_0.011367385275661945|-1025.4033|
|loss_20995_0.01238981168717146|-929.7548|
|loss_20133_0.013162793591618538|-1426.1|
|loss_19985_0.013779347762465477|-1127.3888|

## 6.16
+ 仔细思考了一下，单目相机直接从照片序列识别完整相对运动状态好像有点难
+ 考虑用照片数据集预训练CNN识别门的相对位置和速度（仅相对位置+速度），然后EKF或GRU解算相对速度、角速度？

## 6.17
整理一下目前的FPV视觉导航方案  
**AlphaPilot: Autonomous Drone Racing**
  + Gate_detector+VIO，采用门角检测补偿高速运动时VIO漂移问题，两者EKF融合估计状态
  + CNN采用5层U-Net，[12, 18, 24, 32, 32]，每层卷积核[3, 3, 3, 5, 7]，VIO采用ROVIO
    + ROVIO论文: Robust visual inertial odometry using a direct EKF-based approach.
    + U-Net论文：U-net: Convolutional networks for biomedical image segmentation.
  + 门检测器工作流程：
    1. 从图片中检测所有门角
    2. 网络学习PAF场对门角进行连接
      + PAF论文：Realtime multi-person 2d pose estimation using part affinity fields
  + 控制量为z方向总推力和角速度，轨迹跟踪位置PD控制，姿态P控制

**Deep Drone Racing: Learning Agile Flight in Dynamic Environments 怎么也是穿越移动的门**
  + ResNet输入归一化rgb图片，输出归一化图像坐标中新目标方向和归一化速度，7个卷积层，3个残差块 + 1个全连接层
    + ResNet论文：Dronet: Learning to fly by driving.
  + 控制器从二维图像坐标目标点根据相机投影光线反向+预测深度$d$的位置，计算三维局部坐标系目标点，$d$与归一化速度成正比，目标速度等于归一化速度*$v_{max}$，最后计算最小颤动轨迹抵达目标
    + 跟踪颤动轨迹论文：Automatic re-initialization and failure recovery for aggressive flight with a monocular vision-based quadrotor.
  + 模仿学习，专家轨迹由Minimum snap trajectory generation and control for quadrotors中的算法生成
  + DAgger方法提高对专家未见数据泛化性
  + **发现只要更改静态门的布局，无人机就能泛化到动态门上**
  + gazebo训练
  + 发现直接端到端控制（控制的是无人机速率）学不出来，得加底层控制器
  + 他们的门也是正弦运动
**Learning High-Speed Flight in the Wild**：
  + 输入深度图
  + 采用gazebo+Unity渲染，flightmare控制

