import numpy as np
import pandas as pd

# --- PHYSICS PARAMETERS ---
W = 1.5       # Coupling strength (Energy mismatch boundary)
hbar = 1.0    # Normalized Planck constant
kappa = 2.0   # Decay constant of the evanescent tail (PAni)
k = 5.0       # Wavevector of the propagating Bloch wave (TiO2)

# --- GRID RESOLUTION ---
# You hypothesized 0.05 step changes. We enforce that here.
step_size = 0.05

# Define the 1D arrays for Space (X) and Time (T)
x = np.arange(-3.0, 3.0 + step_size, step_size)
t = np.arange(0.0, 5.0 + step_size, step_size)

# Create a 2D computational grid (Meshgrid)
X, T = np.meshgrid(x, t)

# Initialize an empty matrix of zeros with the exact shape of the grid
Heat = np.zeros_like(X)

# --- EVALUATE THE PIECEWISE AXIOM ---
# Create boolean masks to separate the PAni domain from the TiO2 domain
left_mask = X <= 0
right_mask = X > 0

# Apply the Spatiotemporal Heat Equations to their respective domains
Heat[left_mask] = (np.cos(W * T[left_mask] / hbar)**2) * np.exp(2 * kappa * X[left_mask])
Heat[right_mask] = (np.sin(W * T[right_mask] / hbar)**2) * np.cos(k * X[right_mask])**2

# --- EXPORT FOR ORIGINLAB (XYZ FORMAT) ---
# Flatten the 2D matrices into 1D columns for XYZ formatting
df = pd.DataFrame({
    'Position_X': X.flatten(),
    'Time_T': T.flatten(),
    'Probability_Density_Z': Heat.flatten()
})

df.to_csv('Spatiotemporal_Heatmap_XYZ.csv', index=False)