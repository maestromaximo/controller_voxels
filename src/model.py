
import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Tuple, List, Optional

import src.parameters as params

@dataclass
class SystemMatrices:
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    C_matrices: Tuple[np.ndarray, np.ndarray] # (C_voxel, C_heater) diagonal matrices

class ThermalModel2D:
    def __init__(self, nx: int, ny: int, coupling_mode: str = 'nearest', config: Optional[dict] = None):
        """
        coupling_mode: 'nearest' (default) or 'distance' (inverse distance decay)
        config: Dictionary containing physical parameters. If None, uses src.parameters defaults.
        """
        self.nx = nx
        self.ny = ny
        self.num_voxels = nx * ny
        self.num_states = 2 * self.num_voxels # [T_1...T_N, E_1...E_N]
        self.coupling_mode = coupling_mode
        
        # Load configuration
        if config is None:
            self.config = {
                'k_neighbor': params.K_NEIGHBOR,
                'kappa': params.KAPPA,
                'k_env': params.K_ENV,
                'c_v': params.VOXEL_HEAT_CAPACITY,
                'c_h': params.HEATER_HEAT_CAPACITY,
                'temp_ambient': params.TEMP_AMBIENT,
                'voxel_side_length': params.VOXEL_SIDE_LENGTH
            }
        else:
            self.config = config

        # Map (x, y) to index
        self.idx_map = np.arange(self.num_voxels).reshape((ny, nx))
        
        self.sys = self._build_matrices()
        self._A_inv = np.linalg.inv(self.sys.A)
        self.selector_T = np.hstack([
            np.eye(self.num_voxels),
            np.zeros((self.num_voxels, self.num_voxels))
        ])
        self._phi_cache = {}
        self._step_response_cache = {}
        
    def _get_voxel_coords(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.nx)

    def _get_neighbors_nearest(self, idx: int) -> List[Tuple[int, float]]:
        """Get 4-connected neighbors with coupling strength."""
        neighbors = []
        cy, cx = self._get_voxel_coords(idx)
        
        k_neighbor = self.config['k_neighbor']

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < self.ny and 0 <= nx < self.nx:
                n_idx = self.idx_map[ny, nx]
                neighbors.append((n_idx, k_neighbor))
        return neighbors
        
    def _get_neighbors_distance(self, idx: int) -> List[Tuple[int, float]]:
        """Get ALL other voxels with 1/r^2 coupling (Gauss's Law approximation)."""
        neighbors = []
        cy, cx = self._get_voxel_coords(idx)
        
        # Base coupling at distance L (nearest neighbor distance)
        # k(d) = K_NEIGHBOR * (L / d)^2
        
        L = self.config['voxel_side_length']
        k_neighbor = self.config['k_neighbor']
        # We want k(L) = K_NEIGHBOR
        # So Constant = K_NEIGHBOR * L^2
        
        for i in range(self.num_voxels):
            if i == idx:
                continue
                
            iy, ix = self._get_voxel_coords(i)
            
            # Euclidean distance in meters
            # grid distance * L
            dist_grid = np.sqrt((cy-iy)**2 + (cx-ix)**2)
            dist_meters = dist_grid * L
            
            # k = k0 * (L/d)^2
            k_val = k_neighbor * (L / dist_meters)**2
            neighbors.append((i, k_val))
            
        return neighbors

    def _build_matrices(self) -> SystemMatrices:
        """Construct A, B, E matrices based on the paper's Appendix."""
        N = self.num_voxels
        
        # Initialize blocks
        Lambda = np.zeros((N, N))       # T -> T (coupling)
        Lambda_prime = np.zeros((N, N)) # T -> T (self loss)
        Psi = np.zeros((N, N))          # E -> T
        Phi = np.zeros((N, N))          # T -> E
        Delta = np.zeros((N, N))        # E -> E
        
        # B matrix blocks
        B_top = np.zeros((N, N))
        B_bot = np.eye(N)
        
        # E vector blocks (environmental)
        E_vec = np.zeros(2 * N)
        
        # Parameters
        kappa = self.config['kappa']
        k_env = self.config['k_env']
        c_v = self.config['c_v']
        c_h = self.config['c_h']
        temp_ambient = self.config['temp_ambient']
        
        # Fill matrices
        for i in range(N):
            if self.coupling_mode == 'nearest':
                neighbors = self._get_neighbors_nearest(i)
            elif self.coupling_mode == 'distance':
                neighbors = self._get_neighbors_distance(i)
            else:
                raise ValueError(f"Unknown coupling mode: {self.coupling_mode}")
            
            sum_k_neighbors = 0.0
            
            # Off-diagonal T->T coupling (Lambda)
            for j_idx, k_val in neighbors:
                Lambda[i, j_idx] = k_val
                sum_k_neighbors += k_val
            
            # Diagonal terms
            # Lambda'_{ii} = -kappa_i - k_i^e - sum_{j!=i} k_{ij}
            Lambda_prime[i, i] = -kappa - k_env - sum_k_neighbors
            
            # E -> T (Psi)
            Psi[i, i] = kappa / c_h
            
            # T -> E (Phi)
            Phi[i, i] = kappa * c_h
            
            # E -> E (Delta)
            Delta[i, i] = -kappa
            
            # Environmental vector E
            E_vec[i] = k_env * temp_ambient
            
        # Assemble A
        A_top = np.hstack([Lambda + Lambda_prime, Psi])
        A_bot = np.hstack([Phi, Delta])
        A = np.vstack([A_top, A_bot])
        
        # Assemble B
        B = np.vstack([B_top, B_bot])
        
        return SystemMatrices(A, B, E_vec, (np.eye(N)*c_v, np.eye(N)*c_h))

    def get_steady_state(self, T_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate steady state state (y_star) and input (u_star) for a desired temperature profile.
        """
        N = self.num_voxels
        A = self.sys.A
        E = self.sys.E
        
        T_star = T_targets.flatten()
        
        # Extract blocks
        A_TL = A[:N, :N] # (L + L')
        A_TR = A[:N, N:] # Psi
        A_BL = A[N:, :N] # Phi
        A_BR = A[N:, N:] # Delta
        
        E_top = E[:N]
        
        # Solve for E_star using upper block
        # A_TR @ E_star = -A_TL @ T_star - E_top
        # A_TR is diagonal, so inversion is easy.
        rhs = -A_TL @ T_star - E_top
        psi_diag = np.diag(A_TR)
        E_star = rhs / psi_diag
        
        # Solve for u_star using lower block
        u_star = - (A_BL @ T_star + A_BR @ E_star)
        
        y_star = np.concatenate([T_star, E_star])
        
        return y_star, u_star

    def design_lqr(self, Q_temp_weight=1.0, Q_energy_weight=0.0, R_weight=0.01):
        """
        Compute LQR Gain matrix K.
        """
        N = self.num_voxels
        
        # Q Matrix
        Q_T = np.eye(N) * Q_temp_weight
        Q_E = np.eye(N) * Q_energy_weight
        Q = scipy.linalg.block_diag(Q_T, Q_E)
        
        # R Matrix
        R = np.eye(N) * R_weight
        
        # Solve CARE
        P = scipy.linalg.solve_continuous_are(self.sys.A, self.sys.B, Q, R)
        
        # K = R^-1 B.T P
        R_inv = np.linalg.inv(R)
        K = R_inv @ self.sys.B.T @ P
        
        return K, P

    @staticmethod
    def bryson_weights(temp_max, energy_max=None, power_max=1.0):
        """
        Compute diagonal weight entries using Bryson's rule.

        Parameters
        ----------
        temp_max : float or array-like
            Maximum allowable temperature deviation(s) in degrees.
        energy_max : float or array-like, optional
            Maximum allowable heater-energy deviation(s). If None, defaults
            to temp_max (assuming unity heat capacity).
        power_max : float or array-like
            Maximum allowable control effort(s), typically heater power.

        Returns
        -------
        tuple
            (Q_T_weight, Q_E_weight, R_weight) where each entry is the
            reciprocal squared magnitude of the corresponding maxima.
        """
        temp_max_arr = np.asarray(temp_max, dtype=float)
        if energy_max is None:
            energy_max_arr = temp_max_arr.copy()
        else:
            energy_max_arr = np.asarray(energy_max, dtype=float)
        power_max_arr = np.asarray(power_max, dtype=float)

        q_temp = 1.0 / np.square(temp_max_arr)
        q_energy = 1.0 / np.square(energy_max_arr)
        r_weight = 1.0 / np.square(power_max_arr)

        # Convert singletons back to scalars for convenience
        if q_temp.shape == ():
            q_temp = float(q_temp)
        if q_energy.shape == ():
            q_energy = float(q_energy)
        if r_weight.shape == ():
            r_weight = float(r_weight)

        return q_temp, q_energy, r_weight

    def _phi(self, horizon: float) -> np.ndarray:
        """
        Matrix exponential Φ(H) = e^{A H}.
        """
        if horizon not in self._phi_cache:
            self._phi_cache[horizon] = scipy.linalg.expm(self.sys.A * horizon)
        return self._phi_cache[horizon]

    def get_step_response_matrix(self, horizon: float) -> np.ndarray:
        """
        Cached step response matrix S = 𝔅 A^{-1} (Φ(H) - I) B.
        """
        if horizon not in self._step_response_cache:
            Phi = self._phi(horizon)
            identity = np.eye(self.sys.A.shape[0])
            self._step_response_cache[horizon] = self.selector_T @ (
                self._A_inv @ ((Phi - identity) @ self.sys.B)
            )
        return self._step_response_cache[horizon]

    def predict_state(self, y_current: np.ndarray, u_current: np.ndarray, horizon: float) -> np.ndarray:
        """
        Implements Eq. (395)-(399) from the manuscript:
            y_traj[t+H] = A^{-1}(Φ(H) - I)(B u + E) + Φ(H) y(t)
        """
        Phi = self._phi(horizon)
        identity = np.eye(self.sys.A.shape[0])
        forcing = self.sys.B @ u_current + self.sys.E
        return self._A_inv @ ((Phi - identity) @ forcing) + Phi @ y_current

    def compute_step_response_matrix(self, horizon: float) -> np.ndarray:
        """
        Implements Eq. (523):
            S = 𝔅 A^{-1} (Φ(H) - I) B
        where 𝔅 selects the temperature components.
        """
        Phi = self._phi(horizon)
        identity = np.eye(self.sys.A.shape[0])
        return self.selector_T @ (self._A_inv @ ((Phi - identity) @ self.sys.B))

    def mpc_step(self,
                 y_current: np.ndarray,
                 u_current: np.ndarray,
                 y_target: np.ndarray,
                 horizon: float,
                 q_diag: np.ndarray,
                 p_max: float,
                 step_response: Optional[np.ndarray] = None,
                 q_matrix: Optional[np.ndarray] = None,
                 qs_solver: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single MPC move following Eqs. (377)-(398):
            Δu = (Q S)^{-1} e_C
            e_C = Q (y* - y_traj[t+H])
        Subject to Δu ∈ [-u(t), p - u(t)].
        """
        if step_response is None:
            step_response = self.get_step_response_matrix(horizon)
        if q_matrix is None:
            q_matrix = np.diag(q_diag)

        y_pred = self.predict_state(y_current, u_current, horizon)
        temp_pred = self.selector_T @ y_pred
        temp_target = self.selector_T @ y_target
        e_C = q_matrix @ (temp_target - temp_pred)

        if qs_solver is not None:
            delta_u = qs_solver @ e_C
        else:
            QS = q_matrix @ step_response
            try:
                delta_u = np.linalg.solve(QS, e_C)
            except np.linalg.LinAlgError:
                delta_u = np.linalg.pinv(QS) @ e_C

        lower_bounds = -u_current
        upper_bounds = p_max - u_current

        # Geometric projection from Appendix: scale along the error-space ray
        # so that (QS)^{-1} e_C lies exactly on the boundary of the constraint box.
        alpha = 1.0
        tol = 1e-9 ##tol is there for numerical stability of floating point arithmetic
        for i in range(len(delta_u)):
            if delta_u[i] > upper_bounds[i] + tol and delta_u[i] > 0:
                ratio = upper_bounds[i] / delta_u[i] if delta_u[i] != 0 else 0.0
                if ratio >= 0:
                    alpha = min(alpha, ratio)
            elif delta_u[i] < lower_bounds[i] - tol and delta_u[i] < 0:
                ratio = lower_bounds[i] / delta_u[i] if delta_u[i] != 0 else 0.0
                if ratio >= 0:
                    alpha = min(alpha, ratio)

        if alpha < 1.0 - tol:
            e_C_star = e_C * alpha
            if qs_solver is not None:
                delta_u = qs_solver @ e_C_star
            else:
                try:
                    delta_u = np.linalg.solve(q_matrix @ step_response, e_C_star)
                except np.linalg.LinAlgError:
                    delta_u = np.linalg.pinv(q_matrix @ step_response) @ e_C_star

        delta_u = np.clip(delta_u, lower_bounds, upper_bounds)

        return delta_u
