import matplotlib.pyplot as plt
import torch.optim as optim
import config as cfg
from airsim_env import AirSimEnv
from nn_utils import DynamicsNN, ReplayBuffer, Scaler, load_model_and_buffer, save_model_and_buffer
from cem_mpc_core import adaptive_cem_mpc_episode

def main():
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # Or 'SimHei' etc.
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("中文字体 WenQuanYi Micro Hei 未找到，将使用默认字体。")

    # 初始化
    airsim_environment = AirSimEnv(cfg)
    
    dynamics_model_nn = DynamicsNN(cfg.STATE_DIM, cfg.ACTION_DIM, cfg.NN_HIDDEN_SIZE).to(cfg.device)
    optimizer_nn = optim.Adam(dynamics_model_nn.parameters(), lr=cfg.LEARNING_RATE)
    
    # 选择是否加载老模型
    LOAD_EXISTING_MODEL = True # 是否加载
    MODEL_BASE_FILENAME = "master"

    
    if LOAD_EXISTING_MODEL:
        cfg.EPISODE_EXPLORE=0
        print(f"Attempting to load model and data from base: {MODEL_BASE_FILENAME}...")
        # The load_model_and_buffer function in nn_utils now returns replay_buffer, state_scaler, action_scaler
        loaded_buffer, loaded_state_scaler, loaded_action_scaler = load_model_and_buffer(
            dynamics_model_nn, 
            MODEL_BASE_FILENAME,
            device=cfg.device
        )
        if loaded_buffer is not None:
            main_replay_buffer = loaded_buffer
        else:
            print("Failed to load buffer, initializing new one.")
            main_replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE)
        
        if loaded_state_scaler is not None and loaded_action_scaler is not None:
            main_state_scaler = loaded_state_scaler
            main_action_scaler = loaded_action_scaler
            scalers_are_fitted = main_state_scaler.fitted # Check if loaded scaler was already fitted
        else:
            print("Failed to load scalers, initializing new ones.")
            main_state_scaler = Scaler(cfg.STATE_DIM).to(cfg.device)
            main_action_scaler = Scaler(cfg.ACTION_DIM).to(cfg.device)
            scalers_are_fitted = False
    else:
        print("Starting fresh: Initializing new replay buffer and scalers.")
        main_replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE)
        main_state_scaler = Scaler(cfg.STATE_DIM).to(cfg.device)
        main_action_scaler = Scaler(cfg.ACTION_DIM).to(cfg.device)
        scalers_are_fitted = False # Scalers are not fitted yet

    # 超参数字典
    cem_hyperparams = {
        'prediction_horizon': cfg.PREDICTION_HORIZON,
        'n_samples': cfg.N_SAMPLES_CEM,
        'n_elites': cfg.N_ELITES_CEM,
        'n_iter': cfg.N_ITER_CEM,
        'initial_std': cfg.INITIAL_STD_CEM,
        'min_std': cfg.MIN_STD_CEM,
        'alpha': cfg.ALPHA_CEM
    }
    
    nn_train_hyperparams = {
        'scaler_refit_frequency': cfg.SCALER_REFIT_FREQUENCY,
        'fit_scaler_subset_size': cfg.FIT_SCALER_SUBSET_SIZE,
        'batch_size': cfg.BATCH_SIZE,
        'epochs_per_step': cfg.NN_TRAIN_EPOCHS_PER_STEP,
        'min_buffer_for_training': cfg.MIN_BUFFER_FOR_TRAINING,
        'episode_explore': cfg.EPISODE_EXPLORE, # Num exploration episodes,
        "action_dim":cfg.ACTION_DIM,
        "state_dim":cfg.STATE_DIM
    }

    mpc_task_params = {
        'waypoint_pass_threshold_y': cfg.WAYPOINT_PASS_THRESHOLD_Y,
        'max_sim_time_per_episode': cfg.MAX_SIM_TIME_PER_EPISODE,
        'dt': cfg.DT,
        'control_max': cfg.ACCEL_MAX,
        'q_state_matrix_gpu':cfg.Q_STATE_COST_MATRIX_GPU,
        'r_control_matrix_gpu':cfg.R_CONTROL_COST_MATRIX_GPU,
        'q_terminal_matrix_gpu':cfg.Q_TERMINAL_COST_MATRIX_GPU
    }

    # 记录列表
    all_episode_steps = []
    all_episode_target_reached_flags = []
    all_episode_avg_losses = []
    # last_episode_detailed_data = None # For detailed logging of the final episode if needed

    # 主循环
    for episode_idx in range(cfg.NUM_EPISODES):
        (trajectory_data, controls_data, times_data, nn_losses_episode,
         target_reached_episode, steps_this_episode, avg_loss_episode,
         scalers_are_fitted_after_ep) = adaptive_cem_mpc_episode(
            episode_num=episode_idx,
            airsim_env=airsim_environment,
            nn_model=dynamics_model_nn,
            optimizer=optimizer_nn,
            replay_buffer=main_replay_buffer,
            state_scaler=main_state_scaler,
            action_scaler=main_action_scaler,
            scalers_fitted_once=scalers_are_fitted, # Pass current fitted status
            cem_hyperparams=cem_hyperparams,
            nn_train_hyperparams=nn_train_hyperparams,
            mpc_params=mpc_task_params
        )
        
        scalers_are_fitted = scalers_are_fitted_after_ep # Update status for next episode

        # Log results for this episode
        all_episode_steps.append(steps_this_episode)
        all_episode_target_reached_flags.append(1 if target_reached_episode else 0)
        all_episode_avg_losses.append(avg_loss_episode)

        # Optional: Store detailed data for the last episode
        # if episode_idx == cfg.NUM_EPISODES - 1:
        #     last_episode_detailed_data = {
        #         "trajectory": trajectory_data, "controls": controls_data,
        #         "times": times_data, "nn_losses": nn_losses_episode
        #     }
        #     # You might want to save this detailed data to a file
        #     with open("last_episode_details.pkl", "wb") as f_detail:
        #         pickle.dump(last_episode_detailed_data, f_detail)
        #     print("Detailed data for the last episode saved.")

    # --- End of Training ---
    print("\n--- Training Finished ---")
    # Here you can add code to plot overall training progress, e.g.,
    # plt.figure(figsize=(12, 8))
    # plt.subplot(3,1,1)
    # plt.plot(all_episode_steps)
    # plt.title("Steps per Episode")
    # plt.subplot(3,1,2)
    # plt.plot(all_episode_target_reached_flags)
    # plt.title("Target Reached (1=Yes, 0=No)")
    # plt.subplot(3,1,3)
    # plt.plot(all_episode_avg_losses)
    # plt.title("Average NN Loss per Episode")
    # plt.tight_layout()
    # plt.show()

    # Save final model, buffer, and scalers
    print("Saving final model, buffer, and scalers...")
    save_model_and_buffer(dynamics_model_nn, main_replay_buffer, main_state_scaler, main_action_scaler, "final_trained_model_data")


if __name__ == "__main__":
    main()