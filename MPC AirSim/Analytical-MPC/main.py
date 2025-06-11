import matplotlib.pyplot as plt
import config as cfg
from airsim_env import AirSimEnv
from cem_mpc_core_analytical import adaptive_cem_mpc_episode

def main():
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # Or 'SimHei' etc.
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("中文字体 WenQuanYi Micro Hei 未找到，将使用默认字体。")

    # 初始化
    airsim_environment = AirSimEnv(cfg)

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

    mpc_task_params = {
        'waypoint_pass_threshold_y': cfg.WAYPOINT_PASS_THRESHOLD_Y,
        'max_sim_time_per_episode': cfg.MAX_SIM_TIME_PER_EPISODE,
        'dt': cfg.DT,
        'control_max': cfg.CONTROL_MAX,
        'control_min': cfg.CONTROL_MIN,
        'q_state_matrix_gpu':cfg.Q_STATE_COST_MATRIX_GPU,
        'r_control_matrix_gpu':cfg.R_CONTROL_COST_MATRIX_GPU,
        'q_terminal_matrix_gpu':cfg.Q_TERMINAL_COST_MATRIX_GPU,
        'static_q_state_matrix_gpu':cfg.STATIC_Q_STATE_COST_MATRIX_GPU,
        'static_r_control_matrix_gpu':cfg.STATIC_R_CONTROL_COST_MATRIX_GPU,
        'static_q_terminal_matrix_gpu':cfg.STATIC_Q_TERMINAL_COST_MATRIX_GPU,
        'action_dim':cfg.ACTION_DIM,
        'state_dim':cfg.STATE_DIM
    }

    # 记录列表
    all_episode_steps = []
    all_episode_target_reached_flags = []
    all_episode_avg_losses = []
    # last_episode_detailed_data = None # For detailed logging of the final episode if needed

    # 主循环
    for episode_idx in range(cfg.NUM_EPISODES):
        (trajectory_data, controls_data, times_data,
         target_reached_episode, steps_this_episode) = adaptive_cem_mpc_episode(
            episode_num=episode_idx,
            airsim_env=airsim_environment,
            cem_hyperparams=cem_hyperparams,
            mpc_params=mpc_task_params
        )

        # Log results for this episode
        all_episode_steps.append(steps_this_episode)
        all_episode_target_reached_flags.append(1 if target_reached_episode else 0)

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

if __name__ == "__main__":
    main()