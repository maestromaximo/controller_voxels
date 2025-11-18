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
    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny
        self.num_voxels = nx * ny
        self.num_states = 2 * self.num_voxels # [T_1...T_N, E_1...E_N]
        
        # Map (x, y) to index
        self.idx_map = np.arange(self.num_voxels).reshape((ny, nx))
        
        self.sys = self._build_matrices()
        
    def _get_neighbors(self, idx: int) -> List[int]:
        """Get 4-connected neighbors for a given voxel index."""
        neighbors = []
        cy, cx = divmod(idx, self.nx)
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < self.ny and 0 <= nx < self.nx:
                neighbors.append(self.idx_map[ny, nx])
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
        k_ij = params.K_NEIGHBOR
        kappa = params.KAPPA
        k_env = params.K_ENV
        c_v = params.VOXEL_HEAT_CAPACITY
        c_h = params.HEATER_HEAT_CAPACITY
        
        # Fill matrices
        for i in range(N):
            neighbors = self._get_neighbors(i)
            
            # Off-diagonal T->T coupling (Lambda)
            for j in neighbors:
                Lambda[i, j] = k_ij
            
            # Diagonal terms
            # From paper: Lambda'_{ii} = -kappa_i - k_i^e - sum_{j!=i} k_{ij}
            # Note: Paper Eq 57 has -k_{ij} in the sum. 
            # If sum(-k(Ti-Tj)) = sum(-k Ti) + sum(k Tj)
            # So diagonal gets -sum(k_ij).
            sum_neighbors = len(neighbors) * k_ij
            Lambda_prime[i, i] = -kappa - k_env - sum_neighbors
            
            # E -> T (Psi)
            # From paper: Psi_{ii} = kappa_i / c_i^*
            Psi[i, i] = kappa / c_h
            
            # T -> E (Phi)
            # From paper: Phi_{ii} = kappa_i * c_i^*  <-- WAIT
            # Checking paper Eq 62: dE/dt = -kappa(E - c^* T)
            # dE/dt = -kappa E + kappa c^* T
            # So coeff of T is +kappa * c^*
            Phi[i, i] = kappa * c_h
            
            # E -> E (Delta)
            # Eq 62: coeff of E is -kappa
            Delta[i, i] = -kappa
            
            # Environmental vector E
            # Eq 57: -k^e(T - T^e) -> term +k^e T^e
            E_vec[i] = k_env * params.TEMP_AMBIENT
            # dE/dt has no environmental constant term (Eq 62)
            
        # Assemble A
        # [ L+L'  Psi ]
        # [ Phi   Delta ]
        A_top = np.hstack([Lambda + Lambda_prime, Psi])
        A_bot = np.hstack([Phi, Delta])
        A = np.vstack([A_top, A_bot])
        
        # Assemble B
        B = np.vstack([B_top, B_bot])
        
        return SystemMatrices(A, B, E_vec, (np.eye(N)*c_v, np.eye(N)*c_h))

    def get_steady_state(self, T_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate steady state state (y_star) and input (u_star) for a desired temperature profile.
        
        We set d(delta_y)/dt = 0. 
        Actually, it's easier to solve the linear system Ax + Bu + E = 0 for the full state.
        However, we have N inputs and 2N equations? 
        No, we specify T (N constraints), and we have N inputs (u) and N internal states (E).
        So unknowns are E_star (N) and u_star (N). 2N unknowns.
        Equations are:
        1. 0 = (A y + B u + E)_upper (N equations)
        2. 0 = (A y + B u + E)_lower (N equations)
        
        y = [T_star; E_star]
        
        From lower block (dE/dt = 0):
        0 = Phi T_star + Delta E_star + u_star
        u_star = -Phi T_star - Delta E_star
               = -kappa c^* T_star + kappa E_star
        
        From upper block (dT/dt = 0):
        0 = (Lambda + Lambda') T_star + Psi E_star + E_vec_top
        
        We can solve for E_star from upper block:
        Psi E_star = - (Lambda + Lambda') T_star - E_vec_top
        E_star = Psi^{-1} [ - (Lambda + Lambda') T_star - E_vec_top ]
        
        Then plug into lower block to get u_star.
        """
        N = self.num_voxels
        A = self.sys.A
        E = self.sys.E
        
        T_star = T_targets.flatten()
        
        # Extract blocks
        # A = [[A_TL, A_TR], [A_BL, A_BR]]
        A_TL = A[:N, :N] # (L + L')
        A_TR = A[:N, N:] # Psi
        A_BL = A[N:, :N] # Phi
        A_BR = A[N:, N:] # Delta
        
        E_top = E[:N]
        
        # Solve for E_star using upper block
        # A_TR @ E_star = -A_TL @ T_star - E_top
        # A_TR is diagonal, so inversion is easy.
        # Psi_{ii} = kappa / c_h
        rhs = -A_TL @ T_star - E_top
        
        # Invert A_TR (Psi)
        # Psi_inv_{ii} = c_h / kappa
        psi_diag = np.diag(A_TR)
        E_star = rhs / psi_diag
        
        # Solve for u_star using lower block
        # 0 = A_BL @ T_star + A_BR @ E_star + u_star
        # u_star = - (A_BL @ T_star + A_BR @ E_star)
        # Note: lower block of E vec is 0
        u_star = - (A_BL @ T_star + A_BR @ E_star)
        
        y_star = np.concatenate([T_star, E_star])
        
        return y_star, u_star

    def design_lqr(self, Q_temp_weight=1.0, Q_energy_weight=0.0, R_weight=0.01):
        """
        Compute LQR Gain matrix K.
        J = integral( y.T Q y + u.T R u )
        """
        N = self.num_voxels
        
        # Q Matrix
        # Weight on Temperature error
        Q_T = np.eye(N) * Q_temp_weight
        # Weight on Energy error (usually small or zero)
        Q_E = np.eye(N) * Q_energy_weight
        Q = scipy.linalg.block_diag(Q_T, Q_E)
        
        # R Matrix
        R = np.eye(N) * R_weight
        
        # Solve CARE
        # A.T P + P A - P B R^-1 B.T P + Q = 0
        P = scipy.linalg.solve_continuous_are(self.sys.A, self.sys.B, Q, R)
        
        # K = R^-1 B.T P
        R_inv = np.linalg.inv(R)
        K = R_inv @ self.sys.B.T @ P
        
        return K, P


