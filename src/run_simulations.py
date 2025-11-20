import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import math
from src.model import ThermalModel2D
import src.parameters as params

def run_simulation():
    # Create output directory
    os.makedirs("figures_writting/sim_outputs", exist_ok=True)
    
    # Initialize Models
    NX, NY = 5, 5
    model_nearest = ThermalModel2D(NX, NY, coupling_mode='nearest')
    model_distance = ThermalModel2D(NX, NY, coupling_mode='distance')
    
    N = NX * NY
    
    # LQR Design (Same weights for both)
    Q_T_W, Q_E_W, R_W = 100.0, 0.1, 0.001
    K_nearest, _ = model_nearest.design_lqr(Q_T_W, Q_E_W, R_W)
    K_distance, _ = model_distance.design_lqr(Q_T_W, Q_E_W, R_W)
    MPC_HORIZON = 1
    
    # Common Simulation Settings
    dt = 0.05 
    duration = 1500.0
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)
    
    def simulate(model, K, y0, T_target_map, saturation=True):
        """
        Simulate closed-loop dynamics, returning both saturated (applied)
        and unsaturated LQR inputs.
        """
        y_star, u_star = model.get_steady_state(T_target_map)
        y_current = y0.copy()
        y_hist = np.zeros((steps, 2*N))
        u_hist_sat = np.zeros((steps, N))
        u_hist_unsat = np.zeros((steps, N))
        
        for i in range(steps):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            
            # Unconstrained LQR command
            u_unsat = u_star + delta_u
            # Applied command (possibly saturated)
            if saturation:
                u_applied = np.clip(u_unsat, 0.0, params.P_MAX)
            else:
                u_applied = u_unsat
            
            y_hist[i] = y_current
            u_hist_sat[i] = u_applied
            u_hist_unsat[i] = u_unsat
            
            # Integrate using applied input
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
            
        return time, y_hist, u_hist_sat, u_hist_unsat, y_star, u_star

    def simulate_mpc(model, y0, T_target_map, horizon=MPC_HORIZON, q_temp_weight=1.0,
                     steps_override=None, dt_override=None, record_interval=1):
        """
        Simulate MPC closed-loop using the formulation from src.model.
        """
        y_star, u_star = model.get_steady_state(T_target_map)
        local_steps = steps_override if steps_override is not None else steps
        local_dt = dt_override if dt_override is not None else dt
        sample_interval = max(1, int(record_interval))
        sample_count = math.ceil(local_steps / sample_interval)
        local_time = np.zeros(sample_count)

        y_current = y0.copy()
        # u_current = np.zeros(N)
        u_current = np.clip(u_star.copy(), 0.0, params.P_MAX)
        y_hist = np.zeros((sample_count, 2 * N))
        u_hist = np.zeros((sample_count, N))

        q_diag = np.ones(N) * q_temp_weight
        Q_matrix = np.diag(q_diag)
        S_matrix = model.get_step_response_matrix(horizon)
        QS = Q_matrix @ S_matrix
        try:
            QS_solver = np.linalg.inv(QS)
        except np.linalg.LinAlgError:
            QS_solver = np.linalg.pinv(QS)

        sample_idx = 0
        for i in range(local_steps):
            delta_u = model.mpc_step(
                y_current,
                u_current,
                y_star,
                horizon,
                q_diag,
                params.P_MAX,
                step_response=S_matrix,
                q_matrix=Q_matrix,
                qs_solver=QS_solver,
            )
            u_candidate = u_current + delta_u
            u_current = np.clip(u_candidate, 0.0, params.P_MAX)

            if i % sample_interval == 0 and sample_idx < sample_count:
                y_hist[sample_idx] = y_current
                u_hist[sample_idx] = u_current
                local_time[sample_idx] = i * local_dt
                sample_idx += 1

            dy = model.sys.A @ y_current + model.sys.B @ u_current + model.sys.E
            y_current += dy * local_dt

        y_hist[-1] = y_current
        u_hist[-1] = u_current
        local_time[-1] = (local_steps - 1) * local_dt

        return local_time, y_hist, u_hist, y_star

    # Initial Condition (Ambient)
    y0_amb, _ = model_nearest.get_steady_state(np.ones((NY, NX)) * params.TEMP_AMBIENT)
    
    # --- Simulation 1: Nearest Neighbor - Center Step ---
    print("Re-running Standard Sims (Nearest)...")
    T_target_1 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    center_idx = (NY // 2, NX // 2)
    T_target_1[center_idx] = 100.0
    t1, y1, u1_sat, u1_unsat, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_1)
    
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(8, 10))
    center_flat = center_idx[0]*NX + center_idx[1]
    neighbor_flat = center_flat + 1
    ax1a.plot(t1, y1[:, center_flat], label='Center Voxel')
    ax1a.plot(t1, y1[:, neighbor_flat], label='Neighbor Voxel')
    ax1a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax1a.set_title('Sim 1: Nearest Neighbor - Center Step Response')
    ax1a.set_ylabel('Temperature (°C)')
    ax1a.set_xlabel('Time (s)')
    ax1a.legend()
    ax1a.grid(True)
    
    ax1b.plot(t1, u1_sat[:, center_flat], label='Center Input')
    ax1b.plot(t1, u1_sat[:, neighbor_flat], label='Neighbor Input')
    ax1b.set_title('Control Inputs')
    ax1b.set_ylabel('Power (W)')
    ax1b.set_xlabel('Time (s)')
    ax1b.legend()
    ax1b.grid(True)
    fig1.tight_layout()
    fig1.savefig('figures_writting/sim_outputs/sim1_center_step.png')
    plt.close(fig1)

    # MPC counterpart for Sim 1
    t1_mpc, y1_mpc, u1_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_1)
    fig1_mpc, (ax1m_a, ax1m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax1m_a.plot(t1_mpc, y1_mpc[:, center_flat], label='Center Voxel (MPC)')
    ax1m_a.plot(t1_mpc, y1_mpc[:, neighbor_flat], label='Neighbor Voxel (MPC)')
    ax1m_a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax1m_a.set_title('Sim 1 (MPC): Center Step Response')
    ax1m_a.set_ylabel('Temperature (°C)')
    ax1m_a.set_xlabel('Time (s)')
    ax1m_a.legend()
    ax1m_a.grid(True)

    ax1m_b.plot(t1_mpc, u1_mpc[:, center_flat], label='Center Input (MPC)')
    ax1m_b.plot(t1_mpc, u1_mpc[:, neighbor_flat], label='Neighbor Input (MPC)')
    ax1m_b.set_title('MPC Control Inputs')
    ax1m_b.set_ylabel('Power (W)')
    ax1m_b.set_xlabel('Time (s)')
    ax1m_b.legend()
    ax1m_b.grid(True)
    fig1_mpc.tight_layout()
    fig1_mpc.savefig('figures_writting/sim_outputs/sim1_center_step_mpc.png')
    plt.close(fig1_mpc)

    # --- Simulation 2: Nearest Neighbor - Uniform Heating ---
    T_target_2 = np.ones((NY, NX)) * 100.0
    t2, y2, u2_sat, u2_unsat, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_2)
    
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(8, 12))
    # Temperature trajectories
    ax2a.plot(t2, y2[:, 0], label='Corner Voxel')
    ax2a.plot(t2, y2[:, center_flat], label='Center Voxel', linestyle='--')
    ax2a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax2a.set_title('Sim 2: Nearest Neighbor - Uniform Step (20°C -> 100°C)')
    ax2a.set_ylabel('Temperature (°C)')
    ax2a.set_xlabel('Time (s)')
    ax2a.legend()
    ax2a.grid(True)
    
    # Applied (saturated) inputs
    ax2b.plot(t2, u2_sat[:, 0], label='Corner Input (Saturated)')
    ax2b.plot(t2, u2_sat[:, center_flat], label='Center Input (Saturated)')
    ax2b.axhline(params.P_MAX, color='r', linestyle='--', label='Saturation Limit')
    ax2b.set_title('Control Inputs with Saturation')
    ax2b.set_ylabel('Power (W)')
    ax2b.set_xlabel('Time (s)')
    ax2b.legend()
    ax2b.grid(True)

    # Unconstrained LQR inputs (for reference)
    ax2c.plot(t2, u2_unsat[:, 0], label='Corner Input (Unsaturated)')
    ax2c.plot(t2, u2_unsat[:, center_flat], label='Center Input (Unsaturated)')
    ax2c.set_title('Unconstrained LQR Inputs (Not Physically Realizable)')
    ax2c.set_ylabel('Power (W)')
    ax2c.set_xlabel('Time (s)')
    ax2c.legend()
    ax2c.grid(True)

    fig2.tight_layout()
    fig2.savefig('figures_writting/sim_outputs/sim2_uniform_sat.png')
    plt.close(fig2)

    # MPC counterpart for Sim 2
    t2_mpc, y2_mpc, u2_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_2)
    fig2_mpc, (ax2m_a, ax2m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax2m_a.plot(t2_mpc, y2_mpc[:, 0], label='Corner Voxel (MPC)')
    ax2m_a.plot(t2_mpc, y2_mpc[:, center_flat], label='Center Voxel (MPC)', linestyle='--')
    ax2m_a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax2m_a.set_title('Sim 2 (MPC): Uniform Step (20°C -> 100°C)')
    ax2m_a.set_ylabel('Temperature (°C)')
    ax2m_a.set_xlabel('Time (s)')
    ax2m_a.legend()
    ax2m_a.grid(True)

    ax2m_b.plot(t2_mpc, u2_mpc[:, 0], label='Corner Input (MPC)')
    ax2m_b.plot(t2_mpc, u2_mpc[:, center_flat], label='Center Input (MPC)')
    ax2m_b.axhline(params.P_MAX, color='r', linestyle='--', label='Saturation Limit')
    ax2m_b.set_title('MPC Control Inputs with Saturation')
    ax2m_b.set_ylabel('Power (W)')
    ax2m_b.set_xlabel('Time (s)')
    ax2m_b.legend()
    ax2m_b.grid(True)

    fig2_mpc.tight_layout()
    fig2_mpc.savefig('figures_writting/sim_outputs/sim2_uniform_sat_mpc.png')
    plt.close(fig2_mpc)

    # --- Simulation 3: Nearest Neighbor - Gradient ---
    T_target_3 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    for x in range(NX):
        if x < NX // 2:
            T_target_3[:, x] = 100.0
        else:
            T_target_3[:, x] = 20.0
    t3, y3, u3_sat, u3_unsat, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_3)
    
    fig3, axes = plt.subplots(3, 2, figsize=(10, 12))
    T_final = y3[-1, :N].reshape((NY, NX))
    
    # Heatmap Helper
    def plot_heatmap(ax, data, title, cbar_label, vmin=None, vmax=None, cmap='inferno'):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Voxel X')
        ax.set_ylabel('Voxel Y')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        return im

    # Row 1: Temperatures (final and target)
    plot_heatmap(axes[0,0], T_final, 'Final Temp (Nearest)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)

    # Row 2: Applied steady-state power and temperature error
    plot_heatmap(axes[1,0], u3_sat[-1, :].reshape((NY, NX)),
                 'Steady State Power (Saturated)', 'Power (W)',
                 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes[1,1], T_final - T_target_3,
                 'Error (Actual - Target)', 'Error (°C)',
                 -5, 5, cmap='RdBu_r')

    # Row 3: Unconstrained LQR power and clipping map
    plot_heatmap(axes[2,0], u3_unsat[-1, :].reshape((NY, NX)),
                 'Unconstrained LQR Power', 'Power (W)',
                 None, None, cmap='plasma')
    plot_heatmap(axes[2,1], (u3_unsat[-1, :] - u3_sat[-1, :]).reshape((NY, NX)),
                 'Clipping (Unsat - Sat)', 'Power (W)',
                 None, None, cmap='RdBu_r')
    
    fig3.tight_layout()
    fig3.savefig('figures_writting/sim_outputs/sim3_gradient.png')
    plt.close(fig3)

    # MPC counterpart for Sim 3
    t3_mpc, y3_mpc, u3_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_3)
    fig3_mpc, axes_mpc = plt.subplots(3, 2, figsize=(10, 12))
    T_final_mpc = y3_mpc[-1, :N].reshape((NY, NX))

    plot_heatmap(axes_mpc[0,0], T_final_mpc, 'Final Temp (MPC)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_mpc[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_mpc[1,0], u3_mpc[-1, :].reshape((NY, NX)),
                 'Steady State Power (MPC)', 'Power (W)', 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes_mpc[1,1], T_final_mpc - T_target_3,
                 'Error (Actual - Target)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    plot_heatmap(axes_mpc[2,0], u3_mpc[-1, :].reshape((NY, NX)),
                 'MPC Power Map', 'Power (W)', None, None, cmap='plasma')
    plot_heatmap(axes_mpc[2,1], T_final_mpc - T_final,
                 'Temp Diff (MPC - LQR)', 'Temp (°C)', -5, 5, cmap='PRGn')

    fig3_mpc.tight_layout()
    fig3_mpc.savefig('figures_writting/sim_outputs/sim3_gradient_mpc.png')
    plt.close(fig3_mpc)

    # --- Simulation 3b: Long-Term Gradient (30 min) ---
    print("Running Sim 3b: Long-Term Gradient (30 min)...")
    duration_30m = 1800.0
    steps_30m = int(duration_30m / dt)
    # We can't store full history for 30m at 0.05s dt (36000 steps * 50 states) without waste
    # Just run it and keep final state
    
    y_current = y0_amb.copy()
    y_star, u_star = model_nearest.get_steady_state(T_target_3)
    
    for i in range(steps_30m):
        y_tilde = y_current - y_star
        delta_u = -K_nearest @ y_tilde
        u_applied = u_star + delta_u
        u_applied = np.clip(u_applied, 0.0, params.P_MAX)
        dy = model_nearest.sys.A @ y_current + model_nearest.sys.B @ u_applied + model_nearest.sys.E
        y_current += dy * dt

    fig3b, axes = plt.subplots(2, 2, figsize=(10, 8))
    T_final_30m = y_current[:N].reshape((NY, NX))
    
    plot_heatmap(axes[0,0], T_final_30m, 'Final Temp (30 min)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)
    # Recalculate final input for plotting
    y_tilde = y_current - y_star
    u_final_30m = np.clip(u_star - K_nearest @ y_tilde, 0.0, params.P_MAX)
    plot_heatmap(axes[1,0], u_final_30m.reshape((NY, NX)), 'Steady State Power', 'Power (W)', 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes[1,1], T_final_30m - T_target_3, 'Error (30 min)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    
    fig3b.tight_layout()
    fig3b.savefig('figures_writting/sim_outputs/sim3b_gradient_30m.png')
    plt.close(fig3b)

    # MPC counterpart for Sim 3b
    t3b_mpc, y3b_mpc, u3b_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_3,
                                                steps_override=steps_30m, dt_override=dt)
    T_final_mpc_30m = y3b_mpc[-1, :N].reshape((NY, NX))
    fig3b_mpc, axes_b = plt.subplots(2, 2, figsize=(10, 8))
    plot_heatmap(axes_b[0,0], T_final_mpc_30m, 'Final Temp (MPC, 30 min)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_b[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_b[1,0], u3b_mpc[-1, :].reshape((NY, NX)),
                 'Steady State Power (MPC)', 'Power (W)', 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes_b[1,1], T_final_mpc_30m - T_target_3, 'Error (30 min)', 'Error (°C)',
                 -5, 5, cmap='RdBu_r')
    fig3b_mpc.tight_layout()
    fig3b_mpc.savefig('figures_writting/sim_outputs/sim3b_gradient_30m_mpc.png')
    plt.close(fig3b_mpc)

    # --- Simulation 4: Distance-Based Coupling Comparison ---
    print("Running Distance-Based Sims...")
    
    # Sim 4a: Step Response Comparison
    t4, y4, u4_sat, u4_unsat, _, _ = simulate(model_distance, K_distance, y0_amb, T_target_1)
    t4_mpc, y4_mpc, u4_mpc, _ = simulate_mpc(model_distance, y0_amb, T_target_1)
    
    # --- EXTRA SIM: Long Duration Step Response ---
    print("Running Long Duration Sim...")
    duration_long = 1500.0 # 25 minutes
    steps_long = int(duration_long / dt)
    time_long = np.linspace(0, duration_long, steps_long)
    
    # Re-define simulate for long duration (hacky but quick)
    def simulate_long(model, K, y0, T_target_map):
        y_star, u_star = model.get_steady_state(T_target_map)
        y_current = y0.copy()
        y_hist = np.zeros((steps_long, 2*N))
        
        for i in range(steps_long):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            u_applied = u_star + delta_u
            u_applied = np.clip(u_applied, 0.0, params.P_MAX)
            y_hist[i] = y_current
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
        return time_long, y_hist

    t_long_near, y_long_near = simulate_long(model_nearest, K_nearest, y0_amb, T_target_1)
    t_long_dist, y_long_dist = simulate_long(model_distance, K_distance, y0_amb, T_target_1)
    t_long_near_mpc, y_long_near_mpc, _, _ = simulate_mpc(model_nearest, y0_amb, T_target_1,
                                                          steps_override=steps_long, dt_override=dt)
    t_long_dist_mpc, y_long_dist_mpc, _, _ = simulate_mpc(model_distance, y0_amb, T_target_1,
                                                          steps_override=steps_long, dt_override=dt)

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    ax6.plot(t_long_near, y_long_near[:, 0], 'b-', label='Nearest: Corner')
    ax6.plot(t_long_dist, y_long_dist[:, 0], 'r--', label='Distance: Corner')
    ax6.set_title('Long Duration (1500s) Far-Field Response')
    ax6.set_ylabel('Temperature (°C)')
    ax6.set_xlabel('Time (s)')
    ax6.legend()
    ax6.grid(True)
    fig6.tight_layout()
    fig6.savefig('figures_writting/sim_outputs/sim6_long_duration.png')
    plt.close(fig6)

    fig6_mpc, ax6_mpc = plt.subplots(figsize=(8, 6))
    ax6_mpc.plot(t_long_near_mpc, y_long_near_mpc[:, 0], 'b-', label='Nearest: Corner (MPC)')
    ax6_mpc.plot(t_long_dist_mpc, y_long_dist_mpc[:, 0], 'r--', label='Distance: Corner (MPC)')
    ax6_mpc.set_title('MPC Long Duration (1500s) Far-Field Response')
    ax6_mpc.set_ylabel('Temperature (°C)')
    ax6_mpc.set_xlabel('Time (s)')
    ax6_mpc.legend()
    ax6_mpc.grid(True)
    fig6_mpc.tight_layout()
    fig6_mpc.savefig('figures_writting/sim_outputs/sim6_long_duration_mpc.png')
    plt.close(fig6_mpc)

    # --- Simulation 7: The Impossible "Cold Trap" ---
    print("Running Sim 7: Impossible Cold Trap...")
    # Target: Center = 20C, Surroundings = 100C
    T_target_7 = np.ones((NY, NX)) * 100.0
    T_target_7[center_idx] = 20.0

    # Custom simulation to record both saturated and unsaturated inputs
    y_star_7, u_star_7 = model_nearest.get_steady_state(T_target_7)
    y_current = y0_amb.copy()
    y7 = np.zeros((steps, 2 * N))
    u7_sat = np.zeros((steps, N))
    u7_unsat = np.zeros((steps, N))

    for i in range(steps):
        y_tilde = y_current - y_star_7
        delta_u = -K_nearest @ y_tilde

        # Unsaturated LQR command
        u_unsat = u_star_7 + delta_u
        # Applied (saturated) command respecting physical limits
        u_sat = np.clip(u_unsat, 0.0, params.P_MAX)

        y7[i] = y_current
        u7_sat[i] = u_sat
        u7_unsat[i] = u_unsat

        dy = model_nearest.sys.A @ y_current + model_nearest.sys.B @ u_sat + model_nearest.sys.E
        y_current += dy * dt

    t7 = time
    
    fig7, (ax7a, ax7b, ax7c) = plt.subplots(3, 1, figsize=(8, 12))
    # Temperature response (with saturated inputs)
    ax7a.plot(t7, y7[:, center_flat], label='Center Voxel (Target 20°C)')
    ax7a.plot(t7, y7[:, neighbor_flat], label='Neighbor Voxel (Target 100°C)')
    ax7a.axhline(20.0, color='g', linestyle='--', label='Center Target')
    ax7a.axhline(100.0, color='r', linestyle='--', label='Neighbor Target')
    ax7a.set_title('Sim 7: The "Impossible" Cold Center')
    ax7a.set_ylabel('Temperature (°C)')
    ax7a.set_xlabel('Time (s)')
    ax7a.legend()
    ax7a.grid(True)
    
    # Saturated control inputs (applied)
    ax7b.plot(t7, u7_sat[:, center_flat], label='Center Input (Saturated)')
    ax7b.plot(t7, u7_sat[:, neighbor_flat], label='Neighbor Input (Saturated)')
    ax7b.set_ylabel('Power (W)')
    ax7b.set_xlabel('Time (s)')
    ax7b.set_title('Control Inputs with Saturation (Center rails at 0W)')
    ax7b.legend()
    ax7b.grid(True)

    # Unsaturated LQR commands (for reference)
    ax7c.plot(t7, u7_unsat[:, center_flat], label='Center Input (Unsaturated)')
    ax7c.plot(t7, u7_unsat[:, neighbor_flat], label='Neighbor Input (Unsaturated)')
    ax7c.set_ylabel('Power (W)')
    ax7c.set_xlabel('Time (s)')
    ax7c.set_title('Unconstrained LQR Inputs (Not Physically Realizable)')
    ax7c.legend()
    ax7c.grid(True)
    
    fig7.tight_layout()
    fig7.savefig('figures_writting/sim_outputs/sim7_impossible_trap.png')
    plt.close(fig7)

    # MPC counterpart for Sim 7
    t7_mpc, y7_mpc, u7_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_7)
    fig7_mpc, (ax7m_a, ax7m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax7m_a.plot(t7_mpc, y7_mpc[:, center_flat], label='Center Voxel (MPC)')
    ax7m_a.plot(t7_mpc, y7_mpc[:, neighbor_flat], label='Neighbor Voxel (MPC)')
    ax7m_a.axhline(20.0, color='g', linestyle='--', label='Center Target')
    ax7m_a.axhline(100.0, color='r', linestyle='--', label='Neighbor Target')
    ax7m_a.set_title('Sim 7 (MPC): Cold Trap')
    ax7m_a.set_ylabel('Temperature (°C)')
    ax7m_a.set_xlabel('Time (s)')
    ax7m_a.legend()
    ax7m_a.grid(True)

    ax7m_b.plot(t7_mpc, u7_mpc[:, center_flat], label='Center Input (MPC)')
    ax7m_b.plot(t7_mpc, u7_mpc[:, neighbor_flat], label='Neighbor Input (MPC)')
    ax7m_b.set_ylabel('Power (W)')
    ax7m_b.set_xlabel('Time (s)')
    ax7m_b.set_title('MPC Control Inputs')
    ax7m_b.legend()
    ax7m_b.grid(True)

    fig7_mpc.tight_layout()
    fig7_mpc.savefig('figures_writting/sim_outputs/sim7_impossible_trap_mpc.png')
    plt.close(fig7_mpc)

    # --- Simulation 8: Long-Term Stability ---
    print("Running Sim 8: Long-Term Stability (1 Hour)...")
    duration_stability = 3600.0 # 1 hour
    steps_stab = int(duration_stability / dt)
    time_stab = np.linspace(0, duration_stability, steps_stab)
    save_interval = 100
    
    # Striped Pattern (Columns)
    T_target_8 = np.zeros((NY, NX))
    for x in range(NX):
        if x % 2 == 0:
            T_target_8[:, x] = 80.0
        else:
            T_target_8[:, x] = 40.0
                
    # Reuse simulate_long logic but for this duration/target
    def simulate_stability(model, K, y0, T_target_map):
        y_star, u_star = model.get_steady_state(T_target_map)
        y_current = y0.copy()
        y_hist = np.zeros((steps_stab // save_interval, 2*N))
        t_hist = time_stab[::save_interval]
        
        for i in range(steps_stab):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            u_applied = u_star + delta_u
            u_applied = np.clip(u_applied, 0.0, params.P_MAX)
            
            if i % save_interval == 0:
                idx = i // save_interval
                if idx < len(y_hist):
                    y_hist[idx] = y_current
            
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
            
        return t_hist, y_hist

    t8, y8 = simulate_stability(model_nearest, K_nearest, y0_amb, T_target_8)
    
    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(8, 10))
    # Plot a few random voxels
    ax8a.plot(t8, y8[:, 0], label='Corner (Target 80°C)')
    ax8a.plot(t8, y8[:, 1], label='Neighbor (Target 40°C)')
    ax8a.plot(t8, y8[:, 12], label='Center (Target 80°C)')
    ax8a.set_title('Sim 8: Long-Term Stability (1 Hour)')
    ax8a.set_ylabel('Temperature (°C)')
    ax8a.set_xlabel('Time (s)')
    ax8a.legend()
    ax8a.grid(True)
    ax8a.set_ylim(35, 85) # Zoom in to see steadiness
    
    # Plot Error Norm
    T_target_flat = T_target_8.flatten()
    T_hist = y8[:, :N]
    error_norm = np.linalg.norm(T_hist - T_target_flat, axis=1) / np.sqrt(N) # RMS Error
    
    ax8b.plot(t8, error_norm, 'k-')
    ax8b.set_title('RMS Temperature Error over 1 Hour')
    ax8b.set_ylabel('RMS Error (°C)')
    ax8b.set_xlabel('Time (s)')
    ax8b.grid(True)
    
    fig8.tight_layout()
    fig8.savefig('figures_writting/sim_outputs/sim8_stability.png')
    plt.close(fig8)

    # MPC counterpart for Sim 8
    t8_mpc, y8_mpc, _, _ = simulate_mpc(
        model_nearest,
        y0_amb,
        T_target_8,
        steps_override=steps_stab,
        dt_override=dt,
        record_interval=save_interval,
    )

    fig8_mpc, (ax8m_a, ax8m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax8m_a.plot(t8_mpc, y8_mpc[:, 0], label='Corner (MPC)')
    ax8m_a.plot(t8_mpc, y8_mpc[:, 1], label='Neighbor (MPC)')
    ax8m_a.plot(t8_mpc, y8_mpc[:, 12], label='Center (MPC)')
    ax8m_a.set_title('Sim 8 (MPC): Long-Term Stability')
    ax8m_a.set_ylabel('Temperature (°C)')
    ax8m_a.set_xlabel('Time (s)')
    ax8m_a.legend()
    ax8m_a.grid(True)
    ax8m_a.set_ylim(35, 85)

    T_hist_mpc = y8_mpc[:, :N]
    error_norm_mpc = np.linalg.norm(T_hist_mpc - T_target_flat, axis=1) / np.sqrt(N)
    ax8m_b.plot(t8_mpc, error_norm_mpc, 'k-')
    ax8m_b.set_title('MPC RMS Temperature Error (1 Hour)')
    ax8m_b.set_ylabel('RMS Error (°C)')
    ax8m_b.set_xlabel('Time (s)')
    ax8m_b.grid(True)

    fig8_mpc.tight_layout()
    fig8_mpc.savefig('figures_writting/sim_outputs/sim8_stability_mpc.png')
    plt.close(fig8_mpc)

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Center & Neighbor Response
    ax4a.plot(t1, y1[:, center_flat], 'b-', label='Nearest: Center')
    ax4a.plot(t4, y4[:, center_flat], 'r--', label='Distance: Center')
    ax4a.plot(t1, y1[:, neighbor_flat], 'b:', alpha=0.6, label='Nearest: Neighbor')
    ax4a.plot(t4, y4[:, neighbor_flat], 'r:', alpha=0.6, label='Distance: Neighbor')
    
    ax4a.set_title('Sim 4: Center Step Comparison')
    ax4a.set_ylabel('Temperature (°C)')
    ax4a.set_xlabel('Time (s)')
    ax4a.legend()
    ax4a.grid(True)
    
    # Far Field Response
    corner_flat = 0
    ax4b.plot(t1, y1[:, corner_flat], 'b-', label='Nearest: Corner')
    ax4b.plot(t4, y4[:, corner_flat], 'r--', label='Distance: Corner')
    ax4b.set_title('Far-Field Response (Corner Voxel)')
    ax4b.set_ylabel('Temperature (°C)')
    ax4b.set_xlabel('Time (s)')
    ax4b.legend()
    ax4b.grid(True)
    
    fig4.tight_layout()
    fig4.savefig('figures_writting/sim_outputs/sim4_comparison_step.png')
    plt.close(fig4)

    fig4_mpc, (ax4m_a, ax4m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax4m_a.plot(t1_mpc, y1_mpc[:, center_flat], 'b-', label='Nearest (MPC): Center')
    ax4m_a.plot(t4_mpc, y4_mpc[:, center_flat], 'r--', label='Distance (MPC): Center')
    ax4m_a.plot(t1_mpc, y1_mpc[:, neighbor_flat], 'b:', alpha=0.6, label='Nearest (MPC): Neighbor')
    ax4m_a.plot(t4_mpc, y4_mpc[:, neighbor_flat], 'r:', alpha=0.6, label='Distance (MPC): Neighbor')
    ax4m_a.set_title('Sim 4 (MPC): Center Step Comparison')
    ax4m_a.set_ylabel('Temperature (°C)')
    ax4m_a.set_xlabel('Time (s)')
    ax4m_a.legend()
    ax4m_a.grid(True)

    ax4m_b.plot(t1_mpc, y1_mpc[:, 0], 'b-', label='Nearest (MPC): Corner')
    ax4m_b.plot(t4_mpc, y4_mpc[:, 0], 'r--', label='Distance (MPC): Corner')
    ax4m_b.set_title('Far-Field Response (MPC)')
    ax4m_b.set_ylabel('Temperature (°C)')
    ax4m_b.set_xlabel('Time (s)')
    ax4m_b.legend()
    ax4m_b.grid(True)

    fig4_mpc.tight_layout()
    fig4_mpc.savefig('figures_writting/sim_outputs/sim4_comparison_step_mpc.png')
    plt.close(fig4_mpc)
    
    # Sim 4b: Gradient Heatmap Comparison
    t5, y5, u5_sat, u5_unsat, _, _ = simulate(model_distance, K_distance, y0_amb, T_target_3)
    t5_mpc, y5_mpc, u5_mpc, _ = simulate_mpc(model_distance, y0_amb, T_target_3)
    
    fig5, axes = plt.subplots(2, 2, figsize=(10, 8))
    T_final_dist = y5[-1, :N].reshape((NY, NX))
    
    plot_heatmap(axes[0,0], T_final_dist, 'Final Temp (Distance)', 'Temp (°C)', 20, 100)
    
    err_dist = T_final_dist - T_target_3
    plot_heatmap(axes[0,1], err_dist, 'Error (Distance)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    
    err_nearest = y3[-1, :N].reshape((NY, NX)) - T_target_3
    plot_heatmap(axes[1,0], err_nearest, 'Error (Nearest)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    
    diff_map = T_final_dist - y3[-1, :N].reshape((NY, NX))
    plot_heatmap(axes[1,1], diff_map, 'Diff (Distance - Nearest)', 'Temp Diff (°C)', -2, 2, cmap='PRGn')
    
    fig5.tight_layout()
    fig5.savefig('figures_writting/sim_outputs/sim5_comparison_gradient.png')
    plt.close(fig5)

    fig5_mpc, axes_m = plt.subplots(2, 2, figsize=(10, 8))
    T_final_dist_mpc = y5_mpc[-1, :N].reshape((NY, NX))

    plot_heatmap(axes_m[0,0], T_final_dist_mpc, 'Final Temp (Distance, MPC)', 'Temp (°C)', 20, 100)
    err_dist_mpc = T_final_dist_mpc - T_target_3
    plot_heatmap(axes_m[0,1], err_dist_mpc, 'Error (Distance, MPC)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    plot_heatmap(axes_m[1,0], T_final - T_target_3, 'Error (Nearest, LQR)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    diff_map_mpc = T_final_dist_mpc - T_final_mpc
    plot_heatmap(axes_m[1,1], diff_map_mpc, 'Diff (Distance - Nearest, MPC)', 'Temp Diff (°C)', -2, 2, cmap='PRGn')

    fig5_mpc.tight_layout()
    fig5_mpc.savefig('figures_writting/sim_outputs/sim5_comparison_gradient_mpc.png')
    plt.close(fig5_mpc)

    print("Simulations Complete.")

if __name__ == "__main__":
    run_simulation()
