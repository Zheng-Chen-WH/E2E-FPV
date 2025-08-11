# Learning FPV End-to-end Guidance in Dynamic Enviroment from an Incompetent Teacher      
Zheng Chen

------
## Script List
| Filename | Description |
|---|---|
|**SACfD-SD**|The main and most recent folder, integrating SACfD in a single-gate environment.|
| `analytical_model_gpu.py` | A dynamics prediction model based on AirSim Simpleflight dynamics. |
| `CEM_MPC.py` | An MPC (MPPI) optimization tool using the CEM technique. |
| `config.py` | Stores almost all parameters for the algorithm. |
| `env.py` | The interface for algorithm interaction with AirSim, sending move commands to the drone, gate, etc., and retrieving motion states, images, and other data from AirSim. |
| `main.py` | The main algorithm file, interacting with SAC updates, memory, env, MPC, and other components. |
| `model.py` | Defines the neural network structure and forward propagation. |
| `replay_memory.py` | Defines memory-related functions, including RLmemory and DAggermemory. |
| `sac.py` | The location of the SACfD algorithm, containing parameter update logic, and model saving and loading. |
| `test.py` | A file for code testing. |
| `utils.py` | Contains shared utility functions. |
---
## Script Details
### analytical_model_gpu.py
A drone (quadrotor) dynamics simulator implemented using PyTorch.   
**Main Functions and Features**     
+ Batch Processing: The entire simulator is designed to process a large number of simulation samples (specified by `num_samples` or `K`) in parallel on a GPU, making it highly efficient.
+ Quaternion Operations: Includes a set of PyTorch-based `parallel-computable` quaternion functions for handling 3D rotations and attitude representation.
+ Core Dynamics Model (SimpleFlightDynamicsTorch): This class constructs a simplified drone physics model, considering factors like mass, inertia, motor thrust, and aerodynamic drag. It can predict the drone's flight trajectory (position, velocity, attitude, etc.) over a period `DT` by solving Newton-Euler equations based on a given sequence of motor control commands (PWM values).
+ Aerodynamic Drag: Linear drag is calculated using a box model, while rotational drag is ignored.
+ The model includes a simple low-pass filter to smooth the motor PWM values.      
   
**Input and Output**   
Calling the `simulate_horizon` function with batches of initial states, PWM sequences, and a single step duration outputs batches of trajectories composed of state sequences. Each prediction step calls the `self._dynamics_step` function.

### CEM-MPC.py  
This script implements the **MPC algorithm**, specifically using the **Cross-Entropy Method (CEM) + MPPI algorithm** as its core optimizer.    
**Main Functions and Logic**
+ CEM-MPC Controller: The core of the algorithm, used to calculate the control commands for supervising the learning agent.
+ Prediction and Optimization:
    + Objective: The controller optimizes a sequence of control actions over a predefined prediction horizon.
    + Optimization Flow: At each control step, the MPPI algorithm will:
      1. Randomly generate a large number of possible future control sequences.
      2. Use the `drone dynamics model (analytical_model_gpu.py)` to predict the resulting trajectories for each of these control sequences.
      3. Evaluate the quality of each predicted trajectory using a **cost function (cost_function_gpu)**.
      4. Select the best few trajectories with the lowest cost (the so-called "elite trajectories") and use them to update the probability distribution (mean and variance) for the next round of random sampling.
      5. Repeat steps b-d for a number of iterations (`n_iter`) to iteratively find an optimal control sequence.
+ Rolling Horizon: The algorithm executes only the first action of the found optimal control sequence, then observes the drone's actual state and restarts the entire prediction and optimization process. This is core to MPC's robustness against inaccurate models.
+ Dynamic Target Tracking: The controller can track dynamically changing targets, such as a moving gate, by predicting the gate's future position and using it as the optimization target.
+ GPU Parallelization: All trajectory predictions and cost calculations are performed in parallel on the GPU, allowing the algorithm to evaluate a vast number of possibilities in a short time, outperforming traditional MPC.      
   
**Input and Output**   
Calling the `step` function of the CEM_MPC class with an initial state, progress index, and elapsed time outputs the current optimal action (PWM values).

### env.py  
This script defines the interface class (env) for interacting with the AirSim simulation environment.
**Main Functions and Design**
+ Simulation Environment Encapsulation: Wraps the low-level AirSim API calls into a high-level, standard environment interface similar to OpenAI Gym (`reset` and `step` methods).
+ Dynamic Task Scenario:
    + Moving Gate (`_move_door`): Controls the gate to move sinusoidally.
    + Randomized Initial State: On each `reset`, the gate's initial position, motion phase, and the final target point are randomized to enhance the generalization ability of the trained model.
+ Multi-modal State Acquisition:
    + Physical State (`get_drone_state`): Retrieves the detailed physical state of the drone from AirSim and formats it in two ways: a 13D basic state vector (position, velocity, attitude, etc.) for the MPC, and a more complex state vector including relative positions for the reinforcement learning network.
    + Visual Input (`get_img_sequence`): A key feature of the script is its ability to continuously capture a sequence of first-person view images from the drone and preprocess them into a tensor format suitable for ResNet.
+ Control Interface: The `step` function receives low-level motor control signals from external controllers (MPC and neural network).
+ Reward Mechanism (`step`): To guide the training of the reinforcement learning model, a reward function is designed that considers multiple aspects:
    + Progress Reward: Encourages the drone to fly towards the next target point.
    + Cost Penalty: Penalizes excessive control actions, unstable flight attitudes (high angular velocity), and long flight times.
    + Event Reward/Penalty: Gives a large reward for successfully passing through the gate or reaching the final destination, and a large penalty for collisions.
   
**Input and Output**   
Calling the `step` function with PWM values outputs the analytical drone state, image tensor, Q-network state, reward, done flag, task progress, completion info, elapsed time, relative position to the next target, 9D attitude, relative velocity to the next target, and drone angular velocity.

### main.py  
`main.py` is the main file of the project, integrating all modules to implement the overall training scheme.

**Core Workflow**
+ Initialize All Components:
    + Loads all hyperparameter configurations from `config.py` into a dictionary and distributes them to different modules.
    + Creates instances of `env` (AirSim environment), `CEM_MPC` (expert controller), and `SAC`.
    + Prepares multiple replay buffers to store training data from different sources and with different dimensions.
      + Expert Memory: Stores expert demonstration trajectories for both IL and RL.
      + Exploration Memory: Stores data from actions actually executed during the DAgger process for RL.
      + DAgger Memory: A large-capacity buffer to store DAgger expert data for IL.
      + Recent Memory: A small-capacity, fast-updating buffer to store DAgger expert data for IL.
+ Imitation Learning Training Strategy: DAgger (Dataset Aggregation). The DAgger method combines the advantages of expert demonstration and agent self-exploration.

+ Training Loop:
    + Each round consists of 6 episodes, with the first being an expert demonstration and episodes 2-6 being agent exploration.
    1. Expert Demonstration: The script runs CEM-MPC to control the drone. The trajectory flown by the expert is stored in the "expert replay buffer."
    2. Supervised Learning and Exploration Phase: The SAC agent begins to control the drone. At each step, the system does two things simultaneously:
        + The student network outputs a control action based on the current image, which is executed in the environment. This self-exploration experience (state, own action, reward) is stored in the "exploration replay buffer."
        + The CEM-MPC outputs an expert action for the same state. The state encountered by the student and the "correct answer" action given by the expert are stored in two "DAgger replay buffers" (one large, one small).
    3. Hybrid Update: The SACfD agent's neural network is updated using a hybrid loss function that combines:
        + Imitation Learning Loss: Encourages the student's output to be as close as possible to the expert's "correct answer."
        + Reinforcement Learning Loss: Encourages the student to maximize the long-term reward obtained from the environment.
    4. Repeat steps 1-3.
+ Evaluation and Saving: Training is periodically paused to let the student network complete several tasks independently without expert guidance to evaluate its true performance. The best-performing model is saved.

### model.py  
This script defines the architecture of the neural network models used in the project, primarily including the policy network and value network, to serve the Soft Actor-Critic (SAC) algorithm training.

**End-to-End Policy Network Structure**

+ Hierarchical Spatio-temporal Feature Extraction:
    + Spatial Features: Uses a **ResNet (Residual Network)** as an image encoder. For an input set of four consecutive image frames, ResNet extracts key spatial features from each frame in parallel.
    + Temporal Features: Uses a GRU to process the feature vectors extracted from each image frame sequentially over time.
+ Multi-task Learning and Auxiliary Supervision: To help the network better learn and understand the physical world, the model is designed with two auxiliary task heads:
    + ResNet's auxiliary head predicts the drone's relative position/attitude from a single image.
    + GRU's auxiliary head predicts the drone's relative velocity/angular velocity from the feature sequence.
+ Decision Making:
    + The feature vector output by the GRU and the normalized target point position are concatenated and fed into a standard Multi-Layer Perceptron (MLP), which finally outputs the parameters of a Gaussian distribution (mean and standard deviation). The agent samples from this distribution to generate the final control action (quadrotor PWM values).        
    
**Value Evaluation Network Structure**      
  + A standard SAC network, consisting of two independent Q-networks to improve training stability and avoid overestimation of Q-values.    
  + Each Q-network takes a state and an action as input and outputs a scalar value, used to evaluate the "goodness" (i.e., expected return) of performing that action in that state.

### replay_memory.py
Defines two types of replay buffers for storing, managing, and sampling training data.

**ReplayMemory (Standard Reinforcement Learning Replay Buffer)**      
+ Purpose: Designed for the standard **reinforcement learning** process, storing a complete record of the agent's interactions with the environment.
+ Stored Content: Each record is a complete "state-action-reward-next_state" tuple (s, a, r, s', done).
+ Additional Information: Also stores physical quantity labels for multi-task training of the neural network's auxiliary tasks, such as true relative position, attitude, velocity, etc.      

**DAggerMemory (Imitation Learning/DAgger-specific Replay Buffer)**      
+ Purpose: Designed for the imitation learning algorithm.
+ Stored Content: Stores the expert action for a given state and the corresponding auxiliary task labels.
+ Difference from the former: It does not store information like reward (r) and next state (s'), because the goal of imitation learning is to directly learn the mapping from state to expert action, which is a supervised learning problem and does not require a reward signal.

**Common Features**
+ Circular Buffer: Both classes use a fixed-capacity, first-in-first-out circular list as the underlying storage.
+ Random Sampling: Both provide a `sample` method that can randomly draw a batch of data from the buffer and format it into a layout suitable for neural network input.

### sac.py
`sac.py` is the core and soul of the entire SACfD (SAC from Demonstration) learning algorithm, which combines imitation learning with reinforcement learning.

**Overview**  
This script defines the `SAC` class, which acts as the "brain" of the drone, primarily responsible for:
+ Making Decisions: Selecting actions to execute based on visual input and other state information.
+ Learning from Experience: Optimizing its policy by updating its internal neural networks.
+ Adaptively Fusing Two Different Learning Paradigms:
  + Imitation Learning: Learning by mimicking the behavior of the "expert" (the CEM-MPC controller).
  + Reinforcement Learning: Aiming to maximize the total reward obtained from the environment.
  + **Innovation**: A `dynamic weighting system` is designed to automatically adjust the learning strategy based on the agent's own **learning stability** and **uncertainty**.

**Core Components**    
+ Policy Network (`self.policy`)
    + Model Structure: Adopts the ResNet+GRU+MLP architecture defined in `model.py`.
    + Core Function: Receives a sequence of image frames and other state information as input, then outputs a probability distribution over possible actions. The drone's specific flight actions are sampled from this distribution.
    + Multi-task Learning: The network is required to complete auxiliary tasks, predicting the drone's relative position, attitude, velocity, and angular velocity. This design theoretically forces the network to build a more physically-grounded understanding of the visual input, thereby extracting more robust and meaningful features for the main control task.
+ Value Networks (`self.critic` and `self.critic_target`)
    + Model Structure: Uses a standard QNetwork, applying the Double-Q architecture (i.e., training two Q-networks simultaneously) and the target network delayed update bootstrapping method.
    + Core Function: Used to evaluate the quality of actions. Given a state and an action taken in that state, it predicts the expected future cumulative reward (the "Q-value"). Using two Q-networks and taking the minimum of their predictions helps to stabilize the training process and avoid the common problem of overestimating action values in reinforcement learning. `critic_target` is a time-delayed copy that provides a stable target for learning.

**update_parameters Method**    
In this method, all five networks are trained using data from four buffers in a SACfD learning process that combines reinforcement learning and imitation learning.

**1. Data Preparation**     
  + The function samples data from four different replay buffers:
  + `expert_memory`: Contains imperfect demonstration data from the MPC controller.
  + `exploration_memory`: Contains experiences collected by the SAC agent during its autonomous exploration of the environment.
  + `dagger_memory` & `recent_memory`: Used for the DAgger method, storing "state-expert action" pairs, where the state is encountered by the agent itself, but the action is what the MPC expert would have taken in that state.

These data are then intelligently combined into specific batches for updating the different network modules.

**2. Critic Network Update**     
The critic network is trained using standard SAC logic, learning to accurately predict Q-values using a mix of expert data and the agent's own exploration data.

**3. Policy Network Update (Hybrid Learning Core)**     
This is where the innovation lies. The policy network is updated using a composite loss function consisting of three parts:
  + Imitation Learning (IL) Loss: The action output by the policy network is directly compared with the MPC expert's action for the same state. The loss is the difference between them, requiring the policy network to learn the expert's "good habits."
  + Reinforcement Learning (RL) Loss: The standard SAC policy objective. The policy network attempts to select actions that the critic network rates as having a high Q-value (i.e., leading to high future rewards), enabling the agent to discover strategies that may be better than the expert's or to perform well in situations the expert has never encountered.
  + Auxiliary Task Loss: This is the loss from multi-task learning. The network's predictions for position, velocity, etc., are compared with the labels provided by the simulator, requiring the visual feature extractor to learn physically meaningful representations.

**4. Adaptive Weighting Mechanism (Innovation)**      
A dynamic weighting mechanism is used to balance the Imitation Learning (IL) and Reinforcement Learning (RL) losses.
+ Measuring Uncertainty and Accuracy: At each update step, the algorithm calculates two key metrics:
    + Q-Disagreement: The difference between the predictions of the two critic networks. High disagreement means the agent is uncertain about the value of its current policy.
    - Temporal Difference Error (TD-Error, represented here as Q_loss): The magnitude of the critic network's loss. High error means the critic's evaluation is inaccurate.
+ **Dynamic Baseline**: These raw metrics are compared with their evolving baselines from the recent past.
  + Looking at single values of TD-Error and Disagreement is not enough, as their absolute magnitudes change throughout the training process. A value that seems small initially might be large later. Therefore, a dynamic reference point is needed, which is the baseline (`baseline_td` and `baseline_dis`).
+ The dynamic baseline mechanism can be broken down into the following steps:
  1. Windowed Data Collection: The system does not react to every single measurement, as it can be noisy. It uses a sliding window defined by the `baseline_update_window` parameter to collect all `td_error` and `disagreement` values within that window.
  2. Candidate Baseline Calculation: At the end of the window period, the system calculates the average of all collected values to serve as the "candidate baseline" (`candidate_td`, `candidate_dis`), representing the average learning state of the "recent period."
  3. Baseline Update Logic: The baseline is not updated directly but follows a unidirectional update rule. This rule is controlled by `baseline_update_gamma` (e.g., 0.7):
    + Case A: Learning Worsens (Update Upwards): `candidate_td > self.baseline_td`
    + >If the average error in the current window is higher than the historical baseline, it means the agent has encountered new difficulties and learning has become more turbulent. In this case, the baseline's target value is unconditionally updated to this higher candidate value.
    + Case B: Significant Learning Improvement (Update Downwards): `candidate_td < self.baseline_update_gamma * self.baseline_td`
    + >If the average error is much better than the baseline (e.g., below 70% of the baseline), it marks a significant and stable improvement. In this case, we also update the baseline's target value, lowering it to this new candidate value.
    + Case C: Slight Learning Improvement (No Change)
    + >If the error has only slightly decreased (e.g., between 70% and 100% of the baseline), the baseline's target value remains unchanged.
    + >This is the meaning of the unidirectional update. It prevents the RL weight from decreasing as the baseline continuously drops by small amounts. The baseline is lowered only after confirming a substantial improvement. This tends to keep the baseline at a higher level, avoiding an overly low RL weight.
  4. Smooth Baseline Transition: The baseline value does not immediately "jump" to the new target value at the end of the window. Instead, through `delta_baseline_td` and `delta_baseline_dis`, it moves linearly and smoothly towards the target value during the next window period, avoiding drastic fluctuations in learning weights caused by sudden baseline changes.
  5. Final Weight Calculation: With the dynamic baseline, the current learning state is evaluated at each update:
    + Normalization: The current `current_td_error` is divided by the current `self.baseline_td` to get `norm_td`. This ratio is very meaningful:
        + `norm_td > 1`: Indicates poor network learning, exceeding the recent average level.
        + `norm_td < 1`: Indicates "everything is under control," more stable than the recent average level.
    + RL Weight Calculation: The normalized `norm_td` and `norm_dis` are combined (taking the maximum), and the reinforcement learning weight is calculated through an exponential decay function `w_rl = torch.exp(-k_final * hybrid_metric)`.
        + When learning is turbulent (`hybrid_metric > 1`), `w_rl` becomes very small.
        + When learning is stable (`hybrid_metric < 1`), `w_rl` approaches 1.
    + IL Weight Calculation: The imitation learning weight is `w_il = 1 - w_rl`.

### utils.py
Contains a series of utility functions.
1. **Q-Network Update**     
`soft_update` and `hard_update` functions are used to manage parameter synchronization between the main and target networks.
   + `soft_update(target, source, tau)`
     + Function: Soft Update
     + Purpose: This is the core method for updating the **Target Network** parameters in the SAC algorithm. The target network provides a stable Q-value target for training; if it updates too quickly, training can become unstable.
     + How it works: It updates the target network parameters in a smooth fusion manner. The formula is: `target_params = (1 - tau) * target_params + tau * source_params`. Here, `tau` is a very small value (e.g., 0.005), meaning only a small fraction of the main network's parameters are mixed into the target network each time.
   + `hard_update(target, source)`
     + Function: Hard Update
     + Purpose: Used at the beginning of training to copy the main network's parameters to the target network.
2. **Physics-Informed Weighted MSE Loss Function (Project Innovation)**     
These two functions are one of the key innovations in the imitation learning part of this project.
   + **`weighted_mse_loss(y_pred, y_true)`**   
     + Problem Solved: Standard Mean Squared Error (MSE) treats all samples equally. In expert data, critical actions like sharp turns or stops are rare. A model might tend to learn only common actions and ignore these rare but crucial ones, resulting in a model that only outputs average data.
   + How it works:
     1. It calculates the mean of the expert actions `y_true` within the current batch.
     2. It calculates a weight for each sample. The farther an expert action is from the batch mean, the greater its weight.
     3. The original MSE for each sample is multiplied by its corresponding weight, and then the weighted average loss is calculated.
   + Effect: This mechanism forces the model to pay more attention to expert actions that are far from the weighted mean, preventing it from learning only the mean of the action distribution.
   + **`physics_MSE(y_pred, y_true, weighted=False)`**
     + Problem Solved: Directly comparing the four motor PWM values (i.e., actions) via MSE may not accurately reflect the true physical differences. This mismatch can mislead the model's learning.
     + How it works: It uses a `conversion` helper function to transform the four motor PWM signals into more physically meaningful quantities: 1 total thrust and 3 torques (roll, pitch, yaw). It performs this transformation on both the model's predicted action `y_pred` and the expert's true action `y_true`. Then, it calculates the MSE in this physical space of thrust and torques.
     + It also integrates a weighting idea, giving higher weights to samples whose **action magnitude (average PWM value)** deviates from the batch mean.
   + This loss function requires the model to learn how to produce the correct thrust and torques.
3. **Other Mathematical and Utility Functions**
+ `create_log_gaussian`: Calculates the log probability of a value under a given Gaussian distribution. Using logarithms directly in calculations avoids numerical underflow issues caused by very small probability values and is a standard practice in algorithms like SAC.
+ `logsumexp`: A numerically stable way to compute `log(sum(exp(x)))`. Directly calculating `exp(x)` can lead to overflow if `x` is too large; this function avoids this problem through a mathematical trick.
+ `map_value`: A simple linear mapping function that can proportionally map a value from a range `[a, b]` to another range `[c, d]`. This is a general-purpose utility function.
---
## Summary of Innovations:

### I: Solving the End-to-End Drone Navigation Problem for High-Speed Traversal of a Dynamic Gate

**Challenges**
  + Non-static Target: The gate the agent needs to traverse is undergoing sinusoidal motion with randomized parameters.
  + Prediction Capability Required: The agent must be able to predict the future position of the gate to plan a successful traversal trajectory.
  + High Dynamics: The combination of the drone's high-speed motion and the gate's dynamic movement places higher demands on the control system's real-time performance, precision, and robustness.

### II: Adaptive Policy Learning Framework Combining Imitation and Reinforcement Learning
The top-level design of the entire system utilizes **Model Predictive Control (CEM-MPPI)** as an "offline expert" to provide high-quality control sequences. Then, a neural network is trained using reinforcement learning + imitation learning to learn and further optimize control.    
**Advantages**      
+ Combines the Strengths of Both: Imitation learning provides a good initial policy and stability, avoiding the blind exploration of early reinforcement learning. Reinforcement learning endows the agent with the ability to generalize and surpass the expert in situations it has never encountered.
+ Solves the "Cold Start" Problem: Compared to reinforcement learning from scratch, pre-warming with expert data significantly improves learning efficiency and sample utilization.
+ DAgger Paradigm Application: Uses the DAgger (Dataset Aggregation) process, where the learning agent explores the environment, and an expert then labels the new states encountered by the agent.
 
### III: Adaptive Dynamic Weighting Mechanism Based on Learning State Evaluation
A highly original part of the work, designing a system that can self-evaluate its learning state and automatically adjust weights.    
**Core Idea**: Quantify the accuracy and uncertainty of the agent's learning process through two proxy metrics: Temporal Difference Error (TD-Error) and Q-value Disagreement.    
**Key Technical Details**     
  + Dynamic Baseline: Instead of comparing the absolute values of the metrics, they are compared with a slowly changing, unidirectionally updated dynamic baseline. This baseline only decreases when significant learning progress is made but follows immediately when learning worsens, making the judgment very robust.
  - Adaptive Weight Calculation: When the Q-network is accurate, the system increases the weight of reinforcement learning (RL), encouraging the agent to explore more. When the Q-network is inaccurate, the system increases the weight of imitation learning (IL), forcing the agent to revert to the more reliable expert policy.

### IV: End-to-End Control Network Fusing Spatio-temporal Information with Multi-task Learning Supervision
**Core Idea**: Adopts a ResNet+GRU structure to extract spatial and temporal dynamic features from visual input, respectively.    
**Advantages**:
  + Dynamic Prediction: Uses GRU for relative motion state estimation and prediction.
  + Multi-task Learning: Adds auxiliary supervision tasks, requiring a ResNet auxiliary output to predict relative pose and a GRU auxiliary output to predict relative velocity/angular velocity.

### V: Physics-Informed Loss Function for Imitation Learning
**Core Idea**: Transforms the imitation learning loss function into the space of total thrust and tri-axial torques.    
**How it works**:
  + A `conversion` function is designed to convert the four PWM signals into total thrust and tri-axial torques.
  + The loss function calculates the difference between the thrust/torque produced by the model and that produced by the expert, and integrates a weighting mechanism to give higher penalty weights to actions whose magnitude deviates from the batch mean.
---
## Potential Experiments to Conduct

### Main Table for Quantitative Experiments    
| Experiment Item | Notes |Task Success Rate | Traversal Speed | 
|:---:|---|---|---|
| Baseline | • Pure MPC<br>• UZH's paper: Learning High-Level Policies for Model Predictive Control |
| Complete Model| • Still training |
| Pure Imitation Learning | • Set the reinforcement learning part to zero during learning|
| Pure Reinforcement Learning | • Set the imitation learning part to zero during learning |
|Ablation: Multi-task Learning|• Set the auxiliary task loss part to zero during learning|
|Ablation: Physics-Informed Loss?|• Replace the imitation learning part with standard MSE|

### Qualitative Experiments
+ Compare the flight trajectory of the SACfD-trained model with that of a pure imitation learning model to verify smoothness + PWM value changes.
+ Visualize the feature vectors output by ResNet and GRU?
+ (Quantitative) Compare the results of the auxiliary output heads with the labels to verify their ability to extract physical quantities.
---
## Main Figure Draft
![Main Figure Draft](主图草稿.jpg)

**Main Figure References**
+ Vision-Based Deep Reinforcement Learning of UAV Autonomous Navigation Using Privileged Information         
![Vision-Based Deep Reinforcement Learning of UAV Autonomous Navigation Using Privileged Information](1.jpg "Vision-Based Deep Reinforcement Learning of UAV Autonomous Navigation Using Privileged Information")
+ Champion-level drone racing using deep reinforcement learning         
![Champion-level drone racing using deep reinforcement learning](2.jpg "Champion-level drone racing using deep reinforcement learning")
+ Learning High-Speed Flight in the Wild
![Learning High-Speed Flight in the Wild](3.jpg "Learning High-Speed Flight in the Wild")
+ Actor-Critic Model Predictive Control
![Actor-Critic Model Predictive Control](4.jpg "Actor-Critic Model Predictive Control")
+ Demonstrating Agile Flight from Pixels  without State Estimation
![Demonstrating Agile Flight from Pixels  without State Estimation](5.jpg "Demonstrating Agile Flight from Pixels  without State Estimation")

---