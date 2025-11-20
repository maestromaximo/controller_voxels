import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import time
import math
import json
import optuna
from django.conf import settings
from django.utils import timezone
from django.db.models import Value
from django.db.models.functions import Concat
from src.model import ThermalModel2D
import src.parameters as params
from .models import SimulationConfig, SimulationRun, OptimizationJob

OPTIMIZATION_TIME_LIMIT = 180.0  # seconds

def run_simulation_service(run_id, generate_animation=False):
    run = SimulationRun.objects.get(id=run_id)
    config = run.config
    
    # 1. Calculate Derived Parameters
    voxel_volume = config.voxel_size ** 3
    voxel_mass = config.density * voxel_volume
    c_v = voxel_mass * config.specific_heat
    
    cond_neighbor = config.thermal_conductivity * (config.voxel_size**2) / config.voxel_size
    k_neighbor = cond_neighbor / c_v
    
    # Heuristic for heater coupling (from original params.py)
    cond_heater = cond_neighbor * 10.0
    kappa = cond_heater / c_v # Note: Original code used c_v here, but maybe should be c_h? 
    # Checking params.py: KAPPA = COND_HEATER_AIR / VOXEL_HEAT_CAPACITY. Yes, c_v.
    
    k_env = k_neighbor # Assumption from original code
    
    model_config = {
        'k_neighbor': k_neighbor,
        'kappa': kappa,
        'k_env': k_env,
        'c_v': c_v,
        'c_h': config.heater_heat_capacity,
        'temp_ambient': config.temp_ambient,
        'voxel_side_length': config.voxel_size
    }
    
    # 2. Initialize Model
    model = ThermalModel2D(config.nx, config.ny, coupling_mode=config.coupling_mode, config=model_config)
    
    # 3. Design LQR
    K, _ = model.design_lqr(config.q_temp_weight, config.q_energy_weight, config.r_weight)
    
    # 4. Setup Simulation
    dt = config.dt
    steps = int(config.duration / dt)
    time = np.linspace(0, config.duration, steps)
    N = config.nx * config.ny
    
    # Initial Condition (Ambient)
    y0_amb, _ = model.get_steady_state(np.ones((config.ny, config.nx)) * config.temp_ambient)
    y_current = y0_amb.copy()
    
    # Target
    T_target = np.array(config.target_temp_matrix)
    if T_target.shape != (config.ny, config.nx):
        # Fallback if matrix is wrong shape
        T_target = np.ones((config.ny, config.nx)) * config.temp_ambient
        
    y_star, u_star = model.get_steady_state(T_target)
    
    # Storage
    y_hist = np.zeros((steps, 2*N))
    u_hist = np.zeros((steps, N))
    u_hist_unsat = np.zeros((steps, N))
    
    # 5. Run Loop
    for i in range(steps):
        y_tilde = y_current - y_star
        delta_u = -K @ y_tilde
        
        u_unsat = u_star + delta_u
        u_applied = np.clip(u_unsat, 0.0, config.p_max)
        
        y_hist[i] = y_current
        u_hist[i] = u_applied
        u_hist_unsat[i] = u_unsat
        
        dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
        y_current += dy * dt
        
    # 6. Generate Plots
    run_dir = os.path.join(settings.MEDIA_ROOT, 'runs', str(run.id))
    os.makedirs(run_dir, exist_ok=True)
    
    # Plot 1: Temperature Traces (Center, Corner, Neighbor)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    center_idx = (config.ny // 2) * config.nx + (config.nx // 2)
    corner_idx = 0
    
    ax1.plot(time, y_hist[:, center_idx], label='Center')
    ax1.plot(time, y_hist[:, corner_idx], label='Corner')
    
    # Add target lines
    ax1.axhline(T_target.flatten()[center_idx], color='r', linestyle=':', alpha=0.5, label='Center Target')
    
    ax1.set_title('Temperature Response')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)
    
    path1 = os.path.join(run_dir, 'temp_trace.png')
    fig1.savefig(path1)
    plt.close(fig1)
    
    # Plot 2: Input Power (Applied / Saturated)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(time, u_hist[:, center_idx], label='Center Input')
    ax2.plot(time, u_hist[:, corner_idx], label='Corner Input')
    ax2.set_title('Control Inputs')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Power (W)')
    ax2.legend()
    ax2.grid(True)
    
    path2 = os.path.join(run_dir, 'input_trace.png')
    fig2.savefig(path2)
    plt.close(fig2)

    # Plot 2b: Unconstrained Inputs
    fig2b, ax2b = plt.subplots(figsize=(10, 6))
    ax2b.plot(time, u_hist_unsat[:, center_idx], label='Center Input (Unsat)')
    ax2b.plot(time, u_hist_unsat[:, corner_idx], label='Corner Input (Unsat)')
    ax2b.set_title('Unconstrained Control Inputs')
    ax2b.set_xlabel('Time (s)')
    ax2b.set_ylabel('Power (W)')
    ax2b.legend()
    ax2b.grid(True)

    path2b = os.path.join(run_dir, 'input_unsat_trace.png')
    fig2b.savefig(path2b)
    plt.close(fig2b)
    
    # Plot 3: Final Heatmap
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    
    T_history = y_hist[:, :N].reshape((steps, config.ny, config.nx))
    T_final = T_history[-1]
    im1 = ax3a.imshow(T_final, cmap='inferno', origin='upper')
    ax3a.set_title('Final Temperature')
    plt.colorbar(im1, ax=ax3a)
    
    error = T_final - T_target
    im2 = ax3b.imshow(error, cmap='RdBu_r', origin='upper', vmin=-5, vmax=5)
    ax3b.set_title('Final Error')
    plt.colorbar(im2, ax=ax3b)
    
    path3 = os.path.join(run_dir, 'heatmap.png')
    fig3.savefig(path3)
    plt.close(fig3)

    if generate_animation:
        # Animation: Temperature & Error evolution (5 seconds)
        num_frames = min(100, steps) if steps > 1 else 1
        frame_indices = np.linspace(0, steps - 1, num_frames, dtype=int) if num_frames > 1 else [steps - 1]
        interval_ms = 5000 / num_frames if num_frames > 0 else 5000

        err_history = T_history - T_target
        temp_vmin, temp_vmax = np.min(T_history), np.max(T_history)
        err_abs = np.max(np.abs(err_history))
        err_vlim = max(err_abs, 1.0)

        fig_anim, (ax_anim_temp, ax_anim_err) = plt.subplots(1, 2, figsize=(12, 5))
        im_temp = ax_anim_temp.imshow(T_history[frame_indices[0]], cmap='inferno', origin='upper',
                                      vmin=temp_vmin, vmax=temp_vmax)
        ax_anim_temp.set_title('Temperature Evolution')
        plt.colorbar(im_temp, ax=ax_anim_temp)

        im_err = ax_anim_err.imshow(err_history[frame_indices[0]], cmap='RdBu_r', origin='upper',
                                    vmin=-err_vlim, vmax=err_vlim)
        ax_anim_err.set_title('Error Evolution')
        plt.colorbar(im_err, ax=ax_anim_err)

        def update_anim(frame_idx):
            idx = frame_indices[frame_idx]
            im_temp.set_data(T_history[idx])
            ax_anim_temp.set_xlabel(f"t = {time[idx]:.1f} s")
            im_err.set_data(err_history[idx])
            ax_anim_err.set_xlabel(f"t = {time[idx]:.1f} s")
            return im_temp, im_err

        anim = animation.FuncAnimation(
            fig_anim,
            update_anim,
            frames=num_frames,
            interval=interval_ms,
            blit=False
        )

        path_anim = os.path.join(run_dir, 'heatmap_anim.gif')
        fps = max(1, num_frames / 5 if num_frames > 0 else 1)
        anim.save(path_anim, writer=animation.PillowWriter(fps=fps))
        plt.close(fig_anim)
        run.plot_heatmap_anim_path = os.path.join('runs', str(run.id), 'heatmap_anim.gif')
    else:
        run.plot_heatmap_anim_path = None
    
    # 7. Update Run
    run.status = 'completed'
    run.plot_temp_path = os.path.join('runs', str(run.id), 'temp_trace.png')
    run.plot_input_path = os.path.join('runs', str(run.id), 'input_trace.png')
    run.plot_heatmap_path = os.path.join('runs', str(run.id), 'heatmap.png')
    run.plot_input_unsat_path = os.path.join('runs', str(run.id), 'input_unsat_trace.png')
    
    # Calc Stats
    run.rms_error = np.sqrt(np.mean(error**2))
    
    run.save()


def _append_progress(job_id, message):
    OptimizationJob.objects.filter(id=job_id).update(
        progress_log=Concat('progress_log', Value(message + '\n'))
    )


def _ensure_matrix(matrix_value):
    if isinstance(matrix_value, str):
        if not matrix_value:
            return []
        try:
            return json.loads(matrix_value)
        except json.JSONDecodeError:
            return []
    return matrix_value


def simulate_cost(config_snapshot, q_temp_weight, q_energy_weight, r_weight):
    try:
        nx = int(config_snapshot.get('nx', 5))
        ny = int(config_snapshot.get('ny', 5))
        coupling_mode = config_snapshot.get('coupling_mode', 'nearest')
        voxel_size = float(config_snapshot.get('voxel_size', params.VOXEL_SIDE_LENGTH))
        density = float(config_snapshot.get('density', params.DENSITY_AIR))
        specific_heat = float(config_snapshot.get('specific_heat', params.SPECIFIC_HEAT_AIR))
        thermal_conductivity = float(config_snapshot.get('thermal_conductivity', params.THERMAL_CONDUCTIVITY_AIR))
        heater_heat_capacity = float(config_snapshot.get('heater_heat_capacity', params.HEATER_HEAT_CAPACITY))
        temp_ambient = float(config_snapshot.get('temp_ambient', params.TEMP_AMBIENT))
        dt = max(0.02, float(config_snapshot.get('dt', 0.05)))
        duration = float(config_snapshot.get('duration', 600.0))
        eval_duration = min(duration, 300.0)
        steps = max(5, int(eval_duration / dt))
        p_max = float(config_snapshot.get('p_max', params.P_MAX))

        voxel_volume = voxel_size ** 3
        voxel_mass = density * voxel_volume
        c_v = voxel_mass * specific_heat
        cond_neighbor = thermal_conductivity * (voxel_size**2) / voxel_size
        k_neighbor = cond_neighbor / c_v
        cond_heater = cond_neighbor * 10.0
        kappa = cond_heater / c_v
        k_env = k_neighbor

        model_config = {
            'k_neighbor': k_neighbor,
            'kappa': kappa,
            'k_env': k_env,
            'c_v': c_v,
            'c_h': heater_heat_capacity,
            'temp_ambient': temp_ambient,
            'voxel_side_length': voxel_size,
        }

        model = ThermalModel2D(nx, ny, coupling_mode=coupling_mode, config=model_config)

        try:
            K, _ = model.design_lqr(q_temp_weight, q_energy_weight, r_weight)
        except Exception:
            return float('inf')

        ambient_map = np.ones((ny, nx)) * temp_ambient
        y0_amb, _ = model.get_steady_state(ambient_map)
        y_current = y0_amb.copy()

        target_matrix = _ensure_matrix(config_snapshot.get('target_temp_matrix'))
        if not target_matrix:
            T_target = ambient_map
        else:
            T_target = np.array(target_matrix, dtype=float)
            if T_target.shape != (ny, nx):
                T_target = ambient_map

        y_star, u_star = model.get_steady_state(T_target)
        N = nx * ny
        y_hist = np.zeros((steps, 2 * N))

        for i in range(steps):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            u_unsat = u_star + delta_u
            u_applied = np.clip(u_unsat, 0.0, p_max)

            y_hist[i] = y_current
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt

        T_history = y_hist[:, :N]
        target_flat = T_target.flatten()
        error = T_history - target_flat

        tail_len = max(steps // 5, 1)
        tail_error = error[-tail_len:]
        rms_error = np.sqrt(np.mean(np.square(tail_error)))

        # Tiny L2-style regularizer keeps the search from driving weights
        # arbitrarily close to zero when multiple candidates yield identical
        # RMS error. It is negligible compared to the temperature error term,
        # but breaks ties in favor of more balanced gains.
        regularizer = 1e-6 * (q_temp_weight + q_energy_weight + r_weight)
        score = float(rms_error + regularizer)
        if not math.isfinite(score):
            return float('inf')
        return score
    except Exception:
        return float('inf')


def run_optuna_job(job_id):
    try:
        job = OptimizationJob.objects.get(id=job_id)
    except OptimizationJob.DoesNotExist:
        return

    job.status = 'running'
    job.started_at = timezone.now()
    job.progress_log = 'Initializing Tree-structured Parzen Estimator (Optuna)...\n'
    job.save(update_fields=['status', 'started_at', 'progress_log'])

    snapshot = job.config_snapshot
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    start_time = time.time()
    best_score = math.inf

    def objective(trial):
        nonlocal best_score
        elapsed = time.time() - start_time
        if elapsed >= OPTIMIZATION_TIME_LIMIT:
            raise optuna.exceptions.TrialPruned()

        q_t = trial.suggest_float('q_temp_weight', 1e-3, 1e3, log=True)
        q_e = trial.suggest_float('q_energy_weight', 1e-4, 10.0, log=True)
        r = trial.suggest_float('r_weight', 1e-4, 1.0, log=True)

        score = simulate_cost(snapshot, q_t, q_e, r)
        _append_progress(job_id, f"Trial {trial.number}: cost={score:.4f}, qT={q_t:.4f}, qE={q_e:.4f}, R={r:.5f}")

        if score < best_score:
            best_score = score
            OptimizationJob.objects.filter(id=job_id).update(
                best_q_temp=q_t,
                best_q_energy=q_e,
                best_r=r,
                best_score=score
            )
        return score

    try:
        study.optimize(
            objective,
            timeout=OPTIMIZATION_TIME_LIMIT,
            n_trials=100,
            catch=(optuna.exceptions.TrialPruned,)
        )
        job.refresh_from_db()
        if job.best_q_temp is not None:
            _append_progress(job_id, "Optimization completed successfully.")
            OptimizationJob.objects.filter(id=job_id).update(
                status='completed',
                completed_at=timezone.now()
            )
        else:
            _append_progress(job_id, "Optimization ended without finding a better controller.")
            OptimizationJob.objects.filter(id=job_id).update(
                status='failed',
                completed_at=timezone.now()
            )
    except Exception as exc:
        _append_progress(job_id, f"Optimization failed: {exc}")
        OptimizationJob.objects.filter(id=job_id).update(
            status='failed',
            completed_at=timezone.now()
        )

