
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
