# 一种适用于AirSim Simpleflight的简单控制器
## 目标
通过研究simpleflight和airsim的无人机默认物理参数、底层动力学模型与控制逻辑，实现：
+ 在python中通过物理方程实现由目前状态+控制指令预测下一状态
+ 采用神经网络拟合实际下一状态与预测状态的残差
+ 采用模型计算状态+神经网络拟合残差作为MPC算法动力学模型
>支线任务：理解simpleflight如何从高层指令计算底层指令从而理解为什么之前飞得那么奇怪

## simpleflight主要文件和物理参数
>所有参数都是国际单位制

在/media/zheng/A214861F1485F697/Airsim/AirLib/include/vehicles/multirotor/MultiRotorParams.hpp内：
+ linear_drag_coefficient（线阻力系数）=1.3 / 4.0 = 0.325，计算无人机平动空气阻力使用
+ angular_drag_coefficient（角阻力系数）= linear_drag_coefficient
+ restitution（碰撞弹性系数）= 0.55
+ friction（碰撞摩擦系数）= 0.5
+ 怠速油门50%
+ 无人机电机序列：[QuadX电机](http://ardupilot.org/copter/_images/MOTORS_QuadX_QuadPlus.jpg)

           x_axis
        (2)  |   (0)
             |
        --------------y_axis
             |
        (1)  |   (3)

+ 无人机“机身盒”仅用于计算转动惯量，悬臂装在质心，计算电机位置直接用悬臂长*角度；电机视为质点
+ 顺便一提，UE4里那个无人机模型是1m*1m的巨型无人机（现在已经缩放到）...
+ 转动惯量矩阵：$$I_{xx}=m_{body}/12*(y_{body}^2+z_{body}^2)+4*m_{rotor}*(y_{rotor}^2+z_{rotor}^2)$$
yy、zz照此办理
  + Generic（默认机型，可以在settings.json中添加"Params": {"VehicleName": "Flamewheel"}来改变）
    + 无人机尺寸（基于[DJI F450机身](https://artofcircuits.com/product/quadcopter-frame-hj450-with-power-distribution)）：
    + 臂长0.2275m, rotor_z高0.025m（螺旋桨与机身质心z轴偏差值，指向无人机上方）
    + mass（默认总重量）1kg
    + 默认电机质量0.055kg（MT2212 motor）
    + $m_{body}=mass-4*m_{rotor}=0.78kg$
    + 机身中心盒尺寸x=0.18m, y=0.11m, z=0.04m（中心在质心0,0,0处）
在/media/zheng/A214861F1485F697/Airsim/AirLib/include/vehicles/multirotor/RotorParams.hpp内：
+ 逆时针旋转RotorTurningDirectionCCW = -1 （AirSim坐标系z向下，右手定则）
+ 顺时针旋转RotorTurningDirectionCW = 1
+ 推力系数C_T = 0.109919
+ 扭矩系数 C_P = 0.040164
+ 空气密度air_density = 1.225kg/m^3
+ 最大转速max_rpm = 6396.667RPM
+ 螺旋桨直径propeller_diameter = 0.2286m
+ 螺旋桨高度propeller_height = 0.01m
+ 控制信号低通滤波器时间常数 control_signal_filter_tc = 0.005s
+ 每秒最大转数revolutions_per_second = max_rpm/60
+ 最大角速度max_speed = revolutions_per_second * 2 * M_PIf
+ 最大推力max_thrust = 4.179446268N 根据上述值计算
+ 最大力矩max_torque = 0.055562N·m
  $$\text{max\_thrust} = C_T \cdot \text{air\_density} \cdot n^2 \left(\text{propeller\_diameter}\right)^4$$
  $$\text{max\_torque} = \frac{C_P \cdot \text{air\_density} \cdot n^2 \left(\text{propeller\_diameter}\right)^5}{2 \cdot {\pi}}$$


## simpleflight的控制逻辑（by Gemini）
SimpleFlight 的**核心思想**是将控制指令转化为旋翼的推力与反扭矩，再将这些力矩和空气阻力等汇总，最终通过牛顿-欧拉方程更新无人机的运动状态。

整个流程可以概括为以下几个主要阶段：

1. 控制指令输入与预处理
2. 旋翼力/力矩计算
3. 总力和总力矩汇总
4. 运动学状态更新
### 阶段 1: 控制指令输入与预处理

**目标：**从外部API接收控制信号（如桨叶转速），并对其进行初步处理（如滤波）。

**输入：** 通过 AirSim Python API (例如 client.moveByMotorPWMsAsync([pwm0, pwm1, pwm2, pwm3], duration)) 发送的每个旋翼的控制信号（通常是 0-1 的归一化值）。

**涉及文件：**

+ MultirotorApiBase.hpp / MultirotorApiBase.cpp: (隐式) 这是Python客户端与AirSim通信的API层，接收控制指令。
+ MultiRotorPhysicsBody.hpp (updateSensorsAndController() 方法):
  + vehicle_api_->update(): AirSim内部调用此函数来处理API指令，并将其转换为每个旋翼的原始控制信号。
  + rotors_.at(rotor_index).setControlSignal(vehicle_api_->getActuation(rotor_index));: 将原始控制信号传递给每个 RotorActuator。
+ RotorActuator.hpp (setControlSignal() 方法):
  + control_signal_filter_.setInput(Utils::clip(control_signal, 0.0f, 1.0f));: 接收原始控制信号，并将其裁剪到 0-1 范围，然后送入一个一阶低通滤波器。这个滤波器模拟了电机响应的延迟。

**关键公式：**

+ 滤波器: $$
\text{u\_filtered}(t) = \text{u\_filtered}(t - dt) + \left( \text{u\_input} - \text{u\_filtered}(t - dt) \right) \cdot \left(1 - \exp\left(-\frac{dt}{\text{tc}}\right)\right)$$
  + $u\_input$: 输入的原始控制信号 (0-1)。
  + $u\_filtered$: 滤波后的控制信号。
  + $tc$: 滤波时间常数 (在 RotorParams.hpp 中定义)。
  + 用于平滑输入信号，dt越小，tc越大，输出信号与上一时刻控制信号差值越小（更平滑），缩小时间常数可以有更即时的响应
### 阶段 2: 旋翼力/力矩计算

**目标：** 根据滤波后的控制信号和旋翼参数，计算每个旋翼产生的推力 (Thrust_i) 和反扭矩 (Torque_i)。

**涉及文件：**

+ MultiRotorPhysicsBody.hpp (update() 方法):
  + for (...) { getWrenchVertex(vertex_index).update(); }: 循环调用每个旋翼（RotorActuator）的 update() 方法。
+ RotorActuator.hpp (update(), setOutput(), setWrench() 方法):
  + updateEnvironmentalFactors(): 计算当前空气密度与海平面空气密度的比值 air_density_ratio_。
  + setOutput(output_, params_, control_signal_filter_, turning_direction_): 这是核心计算发生的地方。
    + output.control_signal_filtered = control_signal_filter_.getOutput(): 获取滤波后的控制信号。
    + output.speed = sqrt(output.control_signal_filtered * params.max_speed_square): 计算旋翼转速（角速度）。
      + 推力与转速的平方成正比，而推力通常与控制信号近似成线性关系。
    + output.thrust = output.control_signal_filtered * params.max_thrust: 计算旋翼产生的推力。
    + output.torque_scaler = output.control_signal_filtered * params.max_torque * static_cast<int>(turning_direction);: 计算旋翼产生的反扭矩。
  + setWrench(Wrench& wrench): 将计算出的推力和扭矩施加到 PhysicsBody 的总 wrench_ 上。
    + wrench.force = normal * output_.thrust * air_density_ratio_
      + 推力向量的方向由 normal 决定（通常是向上），大小与当前空气密度成正比。
    + wrench.torque = normal * output_.torque_scaler * air_density_ratio_
      + 反扭矩的方向也沿着 normal 向量（即旋翼的旋转轴），大小同样与空气密度成正比。
+ RotorParams.hpp (RotorParams 结构体):
  + 提供了计算推力 (C_T, max_thrust) 和扭矩 (C_P, max_torque) 所需的系数和最大值。
  + max_thrust 和 max_torque 是在 calculateMaxThrust() 中根据 C_T, C_P, air_density, max_rpm, propeller_diameter 预先计算好的。

**关键公式 (在 RotorActuator::setOutput 中实现)：**

+ 滤波后的控制信号: cs_f = control_signal_filter.getOutput()
+ 旋翼转速 (角速度): $$
\text{speed} = \sqrt{\text{cs\_f} \cdot \text{params.max\_speed\_square}}$$
  + 其中 $$\text{params.max\_speed\_square} = \left(\frac{\text{params.max\_rpm}}{60} \cdot 2 \cdot M\pi\right)^2$$
+ 单个旋翼推力: $$\text{Thrust}_i = \text{cs\_f} \cdot \text{params.max\_thrust}$$
  + 其中$$\text{params.max\_thrust} = \text{params.C\_T} \cdot \text{params.air\_density} \cdot \left(\frac{\text{params.max\_rpm}}{60}\right)^2 \cdot \text{params.propeller\_diameter}^4$$
+ 单个旋翼反扭矩: $$\text{Torque}_i = \text{cs\_f} \cdot \text{params.max\_torque} \cdot \text{turning\_direction}$$
  + 其中$$\text{params.max\_torque} = \frac{\text{params.C\_P} \cdot \text{params.air\_density} \cdot \left(\frac{\text{params.max\_rpm}}{60}\right)^2 \cdot \text{params.propeller\_diameter}^5}{2 \cdot \pi}$$
+ 施加的力/扭矩 (在 RotorActuator::setWrench 中):
$$\text{Force\_on\_body}_i = \text{Thrust}_i \cdot \text{normal\_vector} \cdot \text{air\_density\_ratio}$$
$$\text{Torque\_on\_body}_i = \text{Torque}_i \cdot \text{normal\_vector} \cdot \text{air\_density\_ratio}$$
$$\text{air\_density\_ratio} = \frac{\text{current\_air\_density}}{\text{sea\_level\_air\_density}}(来自 Environment)$$ 

### 阶段 3: 总力和总力矩汇总

**目标：** 将所有旋翼的推力/反扭矩以及空气阻力等外部力矩，转换为作用于无人机重心处的总合力 (F_total) 和总合力矩 (\tau_total)。

**涉及文件：**

+ PhysicsBody.hpp (update() 方法):
  + wrench_ = Wrench::zero(): 在每次更新前清零总力矩。
  + for (...) { getWrenchVertex(vertex_index).update(); }: 每个 RotorActuator 和 DragVertex 的 update() 方法会调用它们的 setWrench()，这些力/力矩会累加到 PhysicsBody 的 wrench_ 成员中。
+ MultiRotorPhysicsBody.hpp (createDragVertices() 方法):
  + 定义了用于计算空气阻力的 drag_faces_。
  + 每个 DragVertex 会根据无人机的线速度和角速度计算其贡献的阻力，并累加到 PhysicsBody 的 wrench_ 中。
+ MultiRotorParams.hpp (linear_drag_coefficient, angular_drag_coefficient):
  + 这些系数用于计算空气阻力。

**关键公式 (在 PhysicsBody 和 DragVertex 中实现)：**

+ 总推力 (U1): 作用在机体 Z 轴方向。 U1 = sum(Thrust_i * normal_vector_i) (在机体坐标系下转换为 Z 轴分量)
+ 横滚力矩 (U2): 绕机体 X 轴。 U2 = sum(r_i x Force_on_body_i)_x (其中 r_i 是旋翼位置向量)
+ 俯仰力矩 (U3): 绕机体 Y 轴。 U3 = sum(r_i x Force_on_body_i)_y
+ 偏航力矩 (U4): 绕机体 Z 轴。 U4 = sum(Torque_i) (每个旋翼的反扭矩，考虑方向)
+ 空气阻力: F_drag = -linear_drag_coefficient * Velocity (简化模型)
+ 角阻力: Tau_drag = -angular_drag_coefficient * Angular_Velocity (简化模型)
+ 总合力: F_total = sum(all forces from rotors and drag)
+ 总合力矩: Tau_total = sum(all torques from rotors and drag)
+ 无人机桨叶投影面积：propeller_area = M_PIf * params.rotor_params.propeller_diameter ^ 2
  + 这里是 直径 * 直径 而非 (直径/2)^2，可能是简化的有效面积计算。
+ 螺旋桨侧面（横截面）面积: propeller_xsection = M_PIf * params.rotor_params.propeller_diameter * params.rotor_params.propeller_height
  + 不知道为啥是直径\*pi\*高，阻力面积应该没有pi
+ “机身盒”上下面积：top_bottom_area = params.body_box.x * params.body_box.y;
+ “机身盒”左右面积：left_right_area = params.body_box.x * params.body_box.z;
+ “机身盒”前后面积： front_back_area = params.body_box.y * params.body_box.z;
+ 三轴方向上阻力因数：drag_factor_unit = \[front_back_area + rotors_.size() * propeller_xsection, left_right_area + rotors_.size() * propeller_xsection, top_bottom_area + rotors_.size() * propeller_area\] * params.linear_drag_coefficient / 2;
  + 空气线阻力$$F_{\text{drag}} = \text{DragFactor} \cdot V^2$$其中$$\text{DragFactor} = 0.5 \cdot \rho \cdot A \cdot C_d$$实际上$$\text{params.linear\_drag\_coefficient} = \rho \cdot C_d$$
### 阶段 4: 运动学状态更新

**目标：**根据总合力、总合力矩、无人机质量和惯性，更新无人机的位置、姿态、线速度和角速度。

**涉及文件：**

+ PhysicsBody.hpp (initialize() 方法):
  + mass_, inertia_: 存储无人机的质量和惯性张量。
  + kinematics_: 指向 Kinematics 对象的指针，用于存储和更新无人机的运动学状态。
+ Kinematics.hpp (Kinematics::State 结构体):
  + 定义了无人机的状态：位置 (Pose), 速度 (Twist), 加速度 (Acceleration), 惯性 (Inertia).
+ FastPhysicsEngine.hpp / FastPhysicsEngine.cpp (隐式):
  + 这是 SimpleFlight 实际执行牛顿-欧拉方程积分的物理求解器。它会从 PhysicsBody 获取 mass_, inertia_ 和 wrench_，然后计算加速度，并更新 kinematics_ 对象。
  + static Wrench getBodyWrench(...)：累加每个wrench（顶点）产生的力偶（力force和力矩torque），force转移到世界坐标系下，torque留在机体坐标系下
  + 

**关键公式 (牛顿-欧拉方程，在物理求解器中实现)：**

+ 线加速度: a = F_total / mass + gravity
+ 角加速度: alpha = Inertia_inv * (Tau_total - (avg_angular_angular_velocity × (Inertia * angular_velocity)))
  + Inertia_inv: 惯性张量的逆。
  + 引入了 (angular_velocity × (Inertia * angular_velocity)) 这一项，用于解释和补偿由于刚体自身旋转导致角动量方向变化而产生的惯性力矩。
  + 其中×为叉乘
+ 线速度更新: V_new = V_old + 0.5 * (a_t+a_t+1) * dt Verlet积分，比直接a_t乘dt稳定一些
+ 角速度更新: omega_new = omega_old + 0.5 * (alpha_t+alpha_t+1) * dt
+ 位置更新: P_new = P_old + V_new * dt
+ 姿态更新: Q_new = Q_old * exp(0.5 * omega_new * dt) (使用四元数进行积分)
+ 力矩torque_i(𝜏)=𝑟×𝐹
