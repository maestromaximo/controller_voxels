import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from src.model import ThermalModel2D
import src.parameters as params

def run_simulation():
    # Create output directory
    os.makedirs("figures_writting/sim_outputs", exist_ok=True)
    
    # Initialize Model (5x5 Grid)
    NX, NY = 5, 5
    model = ThermalModel2D(NX, NY)
    N = NX * NY
    
    # Define LQR Controller
    # Weights: Prioritize Temperature tracking, low penalty on inputs
    K, _ = model.design_lqr(Q_temp_weight=100.0, Q_energy_weight=0.1, R_weight=0.001)
    
    # Common Simulation Settings
    dt = 0.05 # seconds
    duration = 60.0 # seconds
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)
    
    def simulate(y0, T_target_map, label, saturation=True):
        # Calculate Steady State Targets
        y_star, u_star = model.get_steady_state(T_target_map)
        
        # Current State
        y_current = y0.copy()
        
        # History
        y_hist = np.zeros((steps, 2*N))
        u_hist = np.zeros((steps, N))
        
        for i in range(steps):
            # Calculate Error
            y_tilde = y_current - y_star
            
            # LQR Feedback
            delta_u = -K @ y_tilde
            
            # Feedforward + Feedback
            u_applied = u_star + delta_u
            
            # Saturation
            if saturation:
                u_applied = np.clip(u_applied, 0.0, params.P_MAX)
            
            # Store
            y_hist[i] = y_current
            u_hist[i] = u_applied
            
            # Integrate Dynamics (Euler)
            # dy/dt = A y + B u + E
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
            
        return time, y_hist, u_hist, y_star, u_star

    # --- Simulation 1: Center Point Heating (Step Response) ---
    print("Running Sim 1: Center Point Step...")
    T_initial_1 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    # Assume heater energy starts in equilibrium with T_initial
    # E = c* T (roughly, neglecting environmental loss for init)
    E_initial_1 = T_initial_1.flatten() * params.HEATER_HEAT_CAPACITY * params.KAPPA / params.KAPPA # approx
    y0_1 = np.concatenate([T_initial_1.flatten(), np.ones(N)*params.TEMP_AMBIENT*params.HEATER_HEAT_CAPACITY*1.2]) # approximate
    # Actually, let's use get_steady_state to find consistent initial conditions for Ambient
    y0_amb, _ = model.get_steady_state(np.ones((NY, NX)) * params.TEMP_AMBIENT)
    
    # Target: Center to 100C, others Ambient
    T_target_1 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    center_idx = (NY // 2, NX // 2)
    T_target_1[center_idx] = 100.0
    
    t1, y1, u1, y_star1, _ = simulate(y0_amb, T_target_1, "center_step")
    
    # Plot 1
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot Center vs Nearest Neighbor
    center_flat_idx = center_idx[0]*NX + center_idx[1]
    neighbor_flat_idx = center_flat_idx + 1 # Right neighbor
    
    ax1a.plot(t1, y1[:, center_flat_idx], label='Center Voxel (Target 100C)', linewidth=2)
    ax1a.plot(t1, y1[:, neighbor_flat_idx], label='Neighbor Voxel (Target 20C)', linewidth=2, linestyle='--')
    ax1a.axhline(100.0, color='k', linestyle=':', alpha=0.5)
    ax1a.axhline(20.0, color='k', linestyle=':', alpha=0.5)
    ax1a.set_title('Simulation 1: Single Voxel Step Response')
    ax1a.set_ylabel('Temperature (°C)')
    ax1a.legend()
    ax1a.grid(True)
    
    ax1b.plot(t1, u1[:, center_flat_idx], label='Center Input Power')
    ax1b.plot(t1, u1[:, neighbor_flat_idx], label='Neighbor Input Power')
    ax1b.set_ylabel('Power (W)')
    ax1b.set_xlabel('Time (s)')
    ax1b.set_title(f'Control Inputs (Max Power = {params.P_MAX}W)')
    ax1b.legend()
    ax1b.grid(True)
    
    fig1.tight_layout()
    fig1.savefig('figures_writting/sim_outputs/sim1_center_step.png')
    plt.close(fig1)

    # --- Simulation 2: Uniform Heating with Saturation ---
    print("Running Sim 2: Uniform Heating...")
    T_target_2 = np.ones((NY, NX)) * 100.0
    
    t2, y2, u2, _, _ = simulate(y0_amb, T_target_2, "uniform_step")
    
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(8, 10))
    # All voxels are identical by symmetry, plot one
    ax2a.plot(t2, y2[:, 0], label='Corner Voxel')
    ax2a.plot(t2, y2[:, center_flat_idx], label='Center Voxel', linestyle='--')
    ax2a.axhline(100.0, color='k', linestyle=':')
    ax2a.set_title('Simulation 2: Uniform Step 20°C -> 100°C')
    ax2a.set_ylabel('Temperature (°C)')
    ax2a.legend()
    ax2a.grid(True)
    
    ax2b.plot(t2, u2[:, 0], label='Corner Input')
    ax2b.plot(t2, u2[:, center_flat_idx], label='Center Input')
    ax2b.axhline(params.P_MAX, color='r', linestyle='--', label='Saturation Limit')
    ax2b.set_ylabel('Power (W)')
    ax2b.set_xlabel('Time (s)')
    ax2b.legend()
    ax2b.grid(True)
    
    fig2.tight_layout()
    fig2.savefig('figures_writting/sim_outputs/sim2_uniform_sat.png')
    plt.close(fig2)
    
    # --- Simulation 3: Gradient Maintenance ---
    print("Running Sim 3: Gradient...")
    T_target_3 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    # Left columns hot, Right columns cold
    for x in range(NX):
        if x < NX // 2:
            T_target_3[:, x] = 100.0
        else:
            T_target_3[:, x] = 20.0
            
    t3, y3, u3, y_star3, u_star3 = simulate(y0_amb, T_target_3, "gradient")
    
    # Plot Heatmaps of Final State vs Target
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Final Temp
    T_final_map = y3[-1, :N].reshape((NY, NX))
    im0 = axes[0,0].imshow(T_final_map, cmap='inferno', vmin=20, vmax=100)
    axes[0,0].set_title('Final Temperature Distribution')
    plt.colorbar(im0, ax=axes[0,0])
    
    # Target Temp
    im1 = axes[0,1].imshow(T_target_3, cmap='inferno', vmin=20, vmax=100)
    axes[0,1].set_title('Target Temperature')
    plt.colorbar(im1, ax=axes[0,1])
    
    # Final Input Power
    u_final_map = u3[-1, :].reshape((NY, NX))
    im2 = axes[1,0].imshow(u_final_map, cmap='plasma', vmin=0, vmax=params.P_MAX)
    axes[1,0].set_title('Steady State Input Power (W)')
    plt.colorbar(im2, ax=axes[1,0])
    
    # Error
    err_map = T_final_map - T_target_3
    im3 = axes[1,1].imshow(err_map, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1,1].set_title('Temperature Error (°C)')
    plt.colorbar(im3, ax=axes[1,1])
    
    fig3.tight_layout()
    fig3.savefig('figures_writting/sim_outputs/sim3_gradient.png')
    plt.close(fig3)

    print("Simulations Complete. Figures saved to figures_writting/sim_outputs/")

if __name__ == "__main__":
    run_simulation()

