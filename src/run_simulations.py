import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
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
    
    # Common Simulation Settings
    dt = 0.05 
    duration = 60.0
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)
    
    def simulate(model, K, y0, T_target_map, saturation=True):
        y_star, u_star = model.get_steady_state(T_target_map)
        y_current = y0.copy()
        y_hist = np.zeros((steps, 2*N))
        u_hist = np.zeros((steps, N))
        
        for i in range(steps):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            u_applied = u_star + delta_u
            if saturation:
                u_applied = np.clip(u_applied, 0.0, params.P_MAX)
            y_hist[i] = y_current
            u_hist[i] = u_applied
            
            # Integrate
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
            
        return time, y_hist, u_hist, y_star, u_star

    # Initial Condition (Ambient)
    y0_amb, _ = model_nearest.get_steady_state(np.ones((NY, NX)) * params.TEMP_AMBIENT)
    
    # --- Simulation 1: Nearest Neighbor - Center Step ---
    print("Re-running Standard Sims (Nearest)...")
    T_target_1 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    center_idx = (NY // 2, NX // 2)
    T_target_1[center_idx] = 100.0
    t1, y1, u1, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_1)
    
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
    
    ax1b.plot(t1, u1[:, center_flat], label='Center Input')
    ax1b.plot(t1, u1[:, neighbor_flat], label='Neighbor Input')
    ax1b.set_title('Control Inputs')
    ax1b.set_ylabel('Power (W)')
    ax1b.set_xlabel('Time (s)')
    ax1b.legend()
    ax1b.grid(True)
    fig1.tight_layout()
    fig1.savefig('figures_writting/sim_outputs/sim1_center_step.png')
    plt.close(fig1)

    # --- Simulation 2: Nearest Neighbor - Uniform Heating ---
    T_target_2 = np.ones((NY, NX)) * 100.0
    t2, y2, u2, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_2)
    
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(8, 10))
    ax2a.plot(t2, y2[:, 0], label='Corner Voxel')
    ax2a.plot(t2, y2[:, center_flat], label='Center Voxel', linestyle='--')
    ax2a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax2a.set_title('Sim 2: Nearest Neighbor - Uniform Step (20°C -> 100°C)')
    ax2a.set_ylabel('Temperature (°C)')
    ax2a.set_xlabel('Time (s)')
    ax2a.legend()
    ax2a.grid(True)
    
    ax2b.plot(t2, u2[:, 0], label='Corner Input')
    ax2b.plot(t2, u2[:, center_flat], label='Center Input')
    ax2b.axhline(params.P_MAX, color='r', linestyle='--', label='Saturation Limit')
    ax2b.set_title('Control Inputs')
    ax2b.set_ylabel('Power (W)')
    ax2b.set_xlabel('Time (s)')
    ax2b.legend()
    ax2b.grid(True)
    fig2.tight_layout()
    fig2.savefig('figures_writting/sim_outputs/sim2_uniform_sat.png')
    plt.close(fig2)

    # --- Simulation 3: Nearest Neighbor - Gradient ---
    T_target_3 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    for x in range(NX):
        if x < NX // 2: T_target_3[:, x] = 100.0
        else: T_target_3[:, x] = 20.0
    t3, y3, u3, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_3)
    
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
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

    plot_heatmap(axes[0,0], T_final, 'Final Temp (Nearest)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)
    plot_heatmap(axes[1,0], u3[-1, :].reshape((NY, NX)), 'Steady State Power', 'Power (W)', 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes[1,1], T_final - T_target_3, 'Error (Actual - Target)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    
    fig3.tight_layout()
    fig3.savefig('figures_writting/sim_outputs/sim3_gradient.png')
    plt.close(fig3)

    # --- Simulation 4: Distance-Based Coupling Comparison ---
    print("Running Distance-Based Sims...")
    
    # Sim 4a: Step Response Comparison
    t4, y4, u4, _, _ = simulate(model_distance, K_distance, y0_amb, T_target_1)
    
    # --- EXTRA SIM: Long Duration Step Response ---
    print("Running Long Duration Sim...")
    duration_long = 600.0 # 10 minutes
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

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    ax6.plot(t_long_near, y_long_near[:, 0], 'b-', label='Nearest: Corner')
    ax6.plot(t_long_dist, y_long_dist[:, 0], 'r--', label='Distance: Corner')
    ax6.set_title('Long Duration (600s) Far-Field Response')
    ax6.set_ylabel('Temperature (°C)')
    ax6.set_xlabel('Time (s)')
    ax6.legend()
    ax6.grid(True)
    fig6.tight_layout()
    fig6.savefig('figures_writting/sim_outputs/sim6_long_duration.png')
    plt.close(fig6)

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
    
    # Sim 4b: Gradient Heatmap Comparison
    t5, y5, u5, _, _ = simulate(model_distance, K_distance, y0_amb, T_target_3)
    
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

    print("Simulations Complete.")

if __name__ == "__main__":
    run_simulation()
