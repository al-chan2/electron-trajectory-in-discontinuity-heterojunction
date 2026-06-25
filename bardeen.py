import numpy as np
import pandas as pd

# --- 1. THE PHYSICS PARAMETERS ---
# We use realistic schematic scaling to make the physics visually obvious
d = 1.5           # We widened the Transformation Zone to make the decay undeniable
k_pani = 8.0      # High frequency Bloch wave in PAni
kappa = 1.5       # Slower decay constant so we can see the exponential curve clearly
k_tio2 = 5.0      # Lower frequency Bloch wave in TiO2 (different effective mass)

# --- THE ENERGY LANDSCAPE (The most crucial addition) ---
E_total = 1.0     # The Total Energy of the electron (Constant)
V_barrier = 2.5   # The height of the V_bi barrier (Notice V_barrier > E_total)
wave_scale = 0.73  # A scaling factor just to make the waves look nice on the graph

# --- 2. DEFINE THE SPATIAL DOMAINS ---
x_pani = np.linspace(-4, 0, 400, endpoint=False)       
x_zone = np.linspace(0, d, 200, endpoint=False)        
x_tio2 = np.linspace(d, d + 4, 400)                    

# --- 3. CALCULATE THE PURE WAVEFUNCTION (Psi) ---
psi_pani = np.cos(k_pani * x_pani)
psi_zone = np.exp(-kappa * x_zone)

transmitted_amplitude = np.exp(-kappa * d)
psi_tio2 = transmitted_amplitude * np.cos(k_tio2 * (x_tio2 - d))

# Combine the raw waves
x_total = np.concatenate((x_pani, x_zone, x_tio2))
psi_raw = np.concatenate((psi_pani, psi_zone, psi_tio2))

# --- 4. CALCULATE PROBABILITY DENSITY (|Psi|^2) ---
# This is what actual publications plot to show electron presence
prob_density = np.abs(psi_raw)**2

# --- 5. BUILD THE ENERGY LANDSCAPE & SHIFT THE WAVES ---
# We create the V(x) barrier
v_total = np.zeros_like(x_total)
v_total[(x_total >= 0) & (x_total <= d)] = V_barrier

# We create the constant E line
e_line = np.full_like(x_total, E_total)

# THE CRITICAL STEP: Shift the waves to ride on top of the Energy Level
# True Plot = E_total + (Raw Wave * Scale)
psi_shifted = E_total + (psi_raw * wave_scale)
prob_shifted = E_total + (prob_density * wave_scale)

# --- 6. EXPORT TO ORIGIN ---
df = pd.DataFrame({
    'Position_x': x_total,
    'Potential_Barrier_V': v_total,
    'Total_Energy_E': e_line,
    'Wavefunction_Shifted': psi_shifted,
    'Probability_Density_Shifted': prob_shifted
})

df.to_csv('True_Quantum_Transformation.csv', index=False)
print("Data generated! Follow the new OriginLab plotting instructions perfectly.")