import numpy as np
import pandas as pd

# --- PHYSICS PARAMETERS ---
W = 1.5             # Coupling strength (Frequency omega = W/hbar)
hbar = 1.0    
kappa = 2.0         # Evanescent decay constant (PAni)
T_1 = 0.3           # Transmission coeff at x=0 (PAni -> Anatase)
T_2 = 0.8           # Transmission coeff at x=1.5 (Anatase -> Rutile internal scattering)

# --- THE WORLDLINE KINEMATICS ---
k_anatase = 8.0     # Heavy m* -> Slow velocity -> Steep trajectory slope
k_rutile = 4.0      # Light m* -> Fast velocity -> Shallow trajectory slope
boundary_x = 1.5    # The physical location of the internal phase-junction

# --- GRID RESOLUTION ---
step_size = 0.05
x = np.arange(-3.0, 3.0 + step_size, step_size)
t = np.arange(0.0, 5.0 + step_size, step_size)
X, T = np.meshgrid(x, t)

# Initialize the master heatmap
Heat_MSS = np.zeros_like(X)

# --- EVALUATE THE 3-REGION AXIOM ---
# 1. PAni Region (x <= 0)
mask_pani = X <= 0
Heat_MSS[mask_pani] = (np.cos(W * T[mask_pani] / hbar)**2) * np.exp(2 * kappa * X[mask_pani])

# 2. Anatase Region (0 < x <= 1.5)
mask_anatase = (X > 0) & (X <= boundary_x)
# The slow wave propagating through the {001} facets
Heat_MSS[mask_anatase] = T_1 * np.cos(k_anatase * X[mask_anatase] - (W/hbar) * T[mask_anatase])**2

# 3. Rutile Region (x > 1.5)
mask_rutile = X > boundary_x
# The phase is mathematically locked at boundary_x to prevent discontinuity
phase_shift = k_anatase * boundary_x
Heat_MSS[mask_rutile] = T_1 * T_2 * np.cos(k_rutile * (X[mask_rutile] - boundary_x) + phase_shift - (W/hbar) * T[mask_rutile])**2

# --- EXPORT TO ORIGINLAB ---
df_mss = pd.DataFrame({
    'Position_X': X.flatten(), 
    'Time_T': T.flatten(), 
    'Probability_Heat': Heat_MSS.flatten()
})

df_mss.to_csv('Heatmap_MSS_PhaseJunction.csv', index=False)
print("MSS Phase-Junction data generated! Import as XYZ Contour into OriginLab.")