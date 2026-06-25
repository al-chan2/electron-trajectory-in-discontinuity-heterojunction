import numpy as np
import pandas as pd

# --- PHYSICS PARAMETERS ---
W = 1.5       # Coupling strength
hbar = 1.0    
kappa = 2.0   # Decay constant (PAni)

# THE CLOCK 4 DELAY PARAMETERS (The Momentum Trajectories)
# A higher k means a tighter wave and a slower group velocity
k_rutile = 4.0      # Fast wave (No CV Delay)
k_anatase = 8.0     # Slow wave confined by {001} facet (Large CV Delay)

# --- GRID RESOLUTION ---
step_size = 0.05
x = np.arange(-3.0, 3.0 + step_size, step_size)
t = np.arange(0.0, 5.0 + step_size, step_size)
X, T = np.meshgrid(x, t)

# --- INITIALIZE HEATMAPS ---
Heat_Rutile = np.zeros_like(X)
Heat_Anatase = np.zeros_like(X)

left_mask = X <= 0
right_mask = X > 0

# 1. PAni Region (Clock 2 - Stationary Standing Wave)
# This stays identical for both, proving the delay only happens in TiO2
PAni_state = (np.cos(W * T[left_mask] / hbar)**2) * np.exp(2 * kappa * X[left_mask])
Heat_Rutile[left_mask] = PAni_state
Heat_Anatase[left_mask] = PAni_state

# 2. TiO2 Region (Clock 4 - Propagating Bloch Wave Momentum Trajectories)
# We map the traveling wave equation: cos(kX - wT)
Heat_Rutile[right_mask] = 0.3 * np.cos(k_rutile * X[right_mask] - (W/hbar) * T[right_mask])**2
Heat_Anatase[right_mask] = 0.3 * np.cos(k_anatase * X[right_mask] - (W/hbar) * T[right_mask])**2

# --- EXPORT TO ORIGINLAB ---
# We flatten the matrices to create an XYZ grid for Origin's Contour Heatmap
df_rutile = pd.DataFrame({'Position_X': X.flatten(), 'Time_T': T.flatten(), 'Heat_Rutile': Heat_Rutile.flatten()})
df_anatase = pd.DataFrame({'Position_X': X.flatten(), 'Time_T': T.flatten(), 'Heat_Anatase': Heat_Anatase.flatten()})

df_rutile.to_csv('Heatmap_Rutile_Fast.csv', index=False)
df_anatase.to_csv('Heatmap_Anatase_Delay.csv', index=False)
print("Data generated! Plot these as XYZ Contour/Heatmaps in OriginLab.")