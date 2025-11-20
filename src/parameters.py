
import numpy as np

# Physical Constants for AIR Voxel System
# Pure Conduction Model (matching paper formulation)
# System Dimensions: 0.3m x 0.3m
# Grid: 5x5 (2D slice)

SYSTEM_SIDE_LENGTH = 0.3 # m
GRID_SIZE = 5
VOXEL_SIDE_LENGTH = SYSTEM_SIDE_LENGTH / GRID_SIZE # 0.06 m

# Material Properties: AIR
DENSITY_AIR = 1.184 # kg/m^3
SPECIFIC_HEAT_AIR = 1005 # J/(kg*K)
THERMAL_CONDUCTIVITY_AIR = 0.026 # W/(m*K)

# Voxel Properties
VOXEL_VOLUME = VOXEL_SIDE_LENGTH ** 3 # m^3
VOXEL_MASS = DENSITY_AIR * VOXEL_VOLUME # kg
VOXEL_HEAT_CAPACITY = VOXEL_MASS * SPECIFIC_HEAT_AIR # J/K (~0.258)

# Heater Properties
# Small heater node suspended in the voxel
HEATER_HEAT_CAPACITY = 1.6 # J/K (Approx 2g of ceramic/metal)

# 1. Neighbor Coupling (k_ij)
# Pure conduction rate between voxel centers
# Conductance = k_th * Area / Length
COND_NEIGHBOR = THERMAL_CONDUCTIVITY_AIR * (VOXEL_SIDE_LENGTH**2) / VOXEL_SIDE_LENGTH # W/K
# k_ij = Conductance / c_i (Units: 1/s)
K_NEIGHBOR = COND_NEIGHBOR / VOXEL_HEAT_CAPACITY 
# Value: (0.026*0.06)/0.258 ~ 0.006 s^-1
# Note: This is SLOW. Thermal equilibrium takes minutes.

# 2. Heater-to-Voxel Coupling (kappa_i)
# Rate of heat transfer from heater to its own voxel air.
# Representing local conduction/micro-convection from the hot element to the air.
# We pick a value faster than neighbor diffusion but physically grounded.
# Assume Conductance ~ 10x neighbor conductance (closer contact)
COND_HEATER_AIR = COND_NEIGHBOR * 10.0 ## 10 is an order of magnitude stronger
KAPPA = COND_HEATER_AIR / VOXEL_HEAT_CAPACITY # ~0.06 s^-1

# 3. Environmental Loss (k^e)
# Loss through the walls (assumed same as neighbor conduction for now)
K_ENV = K_NEIGHBOR # s^-1

# System Limits
P_MAX = 5.0 # Watts
TEMP_AMBIENT = 20.0 # Celsius
