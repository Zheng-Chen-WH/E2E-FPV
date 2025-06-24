MAVLink (Micro Air Vehicle Link) 是一种非常轻量级的、为微型飞行器（无人机）设计的消息协议。 它的主要目标是实现无人机（飞控）、地面控制站（GCS）以及其他机载或地面设备之间的通信。

## 系统 (System)

**定义：** 在MAVLink网络中，一个“系统”通常指一个独立的、可识别的实体或节点。它可以是一个物理设备，也可以是一个逻辑上的单元。

**例子：**
+ 飞行器本身（飞控是其核心）：这是最常见的系统，通常系统ID为1。
+ 地面控制站 (GCS)：例如QGroundControl、Mission Planner等软件，它们会以一个系统ID（例如255）加入网络。
+ 机载伴侣计算机：如树莓派、NVIDIA Jetson等，如果它们直接参与MAVLink通信，也会被视为一个系统。
+ 天线跟踪器：如果它通过MAVLink与GCS或飞行器通信。
+ 云端服务器：如果它通过MAVLink与无人机交互。

**系统ID (System ID / sysid)：** 每个系统在MAVLink网络中都有一个唯一的8位数字ID（1-255）。这个ID用来在消息中标识消息的发送者和预期的接收者（如果是点对点）。
+ 你代码中的体现：
  + source_system=255: 这行代码将你的Python脚本（作为GCS的角色）的系统ID设置为255。
  + self.drone.target_system: 当收到飞控的心跳包后，Pymavlink库会自动解析出飞控的系统ID，并存储在这个变量中。后续你的脚本向飞控发送指令时，就会使用这个ID作为目标系统ID。
## 组件 (Component)

**定义：** 一个“组件”是“系统”内部的一个功能子单元或模块。一个系统可以包含多个组件，每个组件负责一部分特定的功能。

**例子（假设系统是飞行器，系统ID为1）：**

+ 飞控主板 (Autopilot)：通常组件ID为1 (MAV_COMP_ID_AUTOPILOT1)。这是处理飞行控制逻辑的核心。
+ IMU (惯性测量单元)：可能有自己的组件ID，例如MAV_COMP_ID_IMU。
+ GPS模块：也可能有自己的组件ID。
+ 机载摄像头/云台控制器：如果它通过MAVLink通信。
+ 路径规划器模块：如果它作为飞控系统内的一个独立逻辑单元。

**组件ID (Component ID / compid)：** 每个组件在其所属的系统内部也有一个唯一的8位数字ID。这个ID与系统ID结合，可以更精确地定位到消息的来源或目标。
+ 你代码中的体现：
  + source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER: 这行代码将你的Python脚本（GCS角色）的组件ID设置为190，这是一个预定义的代表任务规划器软件的组件ID。
  + self.drone.target_component: 类似地，从飞控的心跳包中解析出的飞控主组件（通常是飞控本身）的组件ID。
## MAVLink的运作原理

理解了系统和组件后，我们就可以更好地理解MAVLink是如何工作的：

### 基于消息的通信 (Message-Based Communication)：

MAVLink的核心是消息 (Message)。所有数据交换都通过结构化的消息包进行。

每个消息都有一个唯一的消息ID (Message ID)，用于标识消息的类型（例如，HEARTBEAT消息ID是0，GLOBAL_POSITION_INT消息ID是33）。

消息体 (Payload) 包含该类型消息的具体数据（例如，GPS消息包含经纬高、速度等信息）。
消息头部 (Header) 则包含了路由信息（源系统ID、源组件ID、目标系统ID、目标组件ID）、消息序列号、消息ID本身、消息体长度等。

### 心跳机制 (Heartbeat Mechanism)：

self.drone.wait_heartbeat(timeout=10) 这句代码就是在等待这个。

MAVLink网络中的每个系统（尤其是飞控和GCS）都会定期（通常1Hz）广播一个HEARTBEAT消息。

**作用：**
+ 存在宣告：告诉网络中的其他成员“我还活着，并且是这个类型的设备”。
+ 基本状态同步：心跳包中包含了发送者的系统ID、组件ID、飞控类型（如PX4, ArduPilot）、机型（如四旋翼、固定翼）、系统状态（如初始化、待机、活动、校准中等）、MAVLink版本等关键信息。
+ 连接检测：如果一个GCS长时间没有收到某个飞控的心跳，就可以认为连接断开或飞控出现问题。
### 消息路由与寻址：

+ 当一个系统（比如你的Python脚本，sysid=255, compid=190）要发送一个指令给飞控（比如sysid=1, compid=1）时，它会在消息头部填入这些源和目标ID。

+ 广播 vs. 定向：
  + 很多遥测数据（如GPS位置、姿态）通常是飞控向网络中广播的（目标系统ID可能设为0，表示所有系统）。任何感兴趣的系统都可以监听并解析这些消息。
  + 控制指令、参数设置等通常是定向的，即明确指定目标系统ID和目标组件ID。
### 命令与应答 (Commands and Acknowledgements)：

MAVLink定义了一套标准的命令消息，如COMMAND_LONG和COMMAND_INT。这些消息可以用来发送各种指令（起飞、降落、前往航点、设置模式等）。

每个命令都有一个对应的命令ID（例如MAV_CMD_NAV_TAKEOFF）。

当飞控接收到一个命令后，通常会回复一个COMMAND_ACK（命令应答）消息，告知命令是已被接受、正在执行、执行成功、失败还是不被支持。这提供了一种可靠的命令执行机制。
### 参数协议 (Parameter Protocol)：

飞控有大量的可配置参数（如PID增益、地理围栏设置等）。
MAVLink提供了一套消息（如PARAM_REQUEST_READ, PARAM_SET, PARAM_VALUE）来读取、设置和列出这些参数。

GCS软件就是通过这个协议来调整飞控参数的。
### 数据流请求 (Data Stream Requests)：

为了有效利用有限的无线带宽，飞控通常不会以最高频率发送所有类型的遥测数据。

GCS或伴侣计算机可以通过REQUEST_DATA_STREAM消息请求飞控以特定的频率发送特定的数据流（例如，高频姿态数据流、中频位置数据流等）。
### 方言 (Dialects) 与消息定义 (Message Definitions)：

MAVLink消息的结构是在XML文件中定义的（例如common.xml定义了所有MAVLink系统都应支持的通用消息）。

不同的项目（如ArduPilot, PX4）可以在通用消息的基础上定义自己的“方言”(Dialect)，即包含特定于该项目的额外消息定义的XML文件（如ardupilotmega.xml）。

Pymavlink这类库会使用这些XML文件来自动生成用于打包和解包消息的代码。
### 传输层无关 (Transport Layer Agnostic)：

MAVLink协议本身只定义消息的格式和交换逻辑，它不关心底层的物理传输方式。
它可以运行在串口（如数传电台）、UDP（如Wi-Fi或以太网）、TCP等多种传输层之上。这使得MAVLink非常灵活。

mavutil.mavlink_connection(self.mavlink_conn_str, ...) 中的self.mavlink_conn_str就指定了传输方式和地址，例如'udp:127.0.0.1:14540'表示通过UDP连接到本地的14540端口。

**总结一下你代码片段中发生的事情，结合MAVLink原理：**

你的Python脚本（系统ID 255，组件ID 190）尝试通过指定的连接字符串（例如UDP端口）连接到一个MAVLink设备。

它开始监听，等待一个心跳包的到来，以确认网络中至少有一个MAVLink设备在线并且愿意通信。

一旦收到心跳包（假设来自系统ID为1，组件ID为1的PX4飞控），Pymavlink就知道了这个飞控的“地址”。

你的脚本随后将self.target_system和self.target_component设置为1和1，这样，当你的脚本后续要发送起飞、降落等指令时，它就知道要把这些指令的目标地址设置为(sysid=1, compid=1)。

MAVLink通过这种结构化的消息传递、明确的寻址（系统/组件ID）以及标准化的核心机制（心跳、命令、参数等），实现了不同无人机相关软硬件之间的有效通信和互操作性。