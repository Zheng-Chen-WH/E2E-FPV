from nn_utils import DynamicsNN, ReplayBuffer, Scaler, load_model_and_buffer, save_model_and_buffer
import torch.optim as optim
import config as cfg
dynamics_model_nn = DynamicsNN(cfg.STATE_DIM, cfg.ACTION_DIM, cfg.NN_HIDDEN_SIZE).to(cfg.device)
optimizer_nn = optim.Adam(dynamics_model_nn.parameters(), lr=cfg.LEARNING_RATE)
loaded_buffer, loaded_state_scaler, loaded_action_scaler = load_model_and_buffer(
            dynamics_model_nn, 
            "master",
            device=cfg.device
        )
for i in range(5):
    state, action, next_state = loaded_buffer.sample(1)
    print(f"\nstate from buffer:[{state[0][0]:.2f}, {state[0][1]:.2f},{state[0][2]:.2f},{state[0][3]:.2f},{state[0][4]:.2f},{state[0][5]:.2f},{state[0][6]:.2f},{state[0][7]:.2f},{state[0][8]:.2f},{state[0][9]:.2f},{state[0][10]:.2f},{state[0][11]:.2f}]")
    print(f"action from buffer:[{action[0][0]:.2f},{action[0][1]:.2f},{action[0][2]:.2f}]")        
    print(f"next_state from buffer:[{next_state[0][0]:.2f},{next_state[0][1]:.2f},{next_state[0][2]:.2f},{next_state[0][3]:.2f},{next_state[0][4]:.2f},{next_state[0][5]:.2f},{next_state[0][6]:.2f},{next_state[0][7]:.2f},{next_state[0][8]:.2f},{next_state[0][9]:.2f},{next_state[0][10]:.2f},{next_state[0][11]:.2f}]")