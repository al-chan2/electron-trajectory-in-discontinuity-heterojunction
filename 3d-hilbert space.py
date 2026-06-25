import numpy as np
import pandas as pd

# --- 1. MESHGRID IN HILBERT SPACE ---
# Instead of a single line, we create a 2D grid of starting parameters.
# L is the Gauge Condition (0 to 1) -> Z axis
# Phi_0 is the initial phase of the incoming wave (0 to 2*pi)
L_vals = np.linspace(0, 1, 200)
Phi_vals = np.linspace(0, 2*np.pi, 200)

L, Phi_0 = np.meshgrid(L_vals, Phi_vals)

# --- 2. THE SINGULARITY DECAY ---
A_initial = 1.0          # PAni state amplitude
A_final = 0.3            # TiO2 state amplitude
phase_twists = 1.0       # The phase rotation induced by the operator

# The exponential amplitude crush (Steepest Descent)
Amplitude = A_initial * np.exp(np.log(A_final) * L)

# The total phase evolution
Phase = Phi_0 + (phase_twists * 2 * np.pi * L)

# --- 3. YOUR EXACT HILBERT AXES ---
Re_Psi = Amplitude * np.cos(Phase)  # X-axis
Im_Psi = Amplitude * np.sin(Phase)  # Y-axis
Gauge_Z = L                         # Z-axis
Heat = Amplitude**2                 # Color Map (Probability Density)

# --- 4. EXPORT TO ORIGINLAB ---
df = pd.DataFrame({
    'Real_Psi_X': Re_Psi.flatten(),
    'Imag_Psi_Y': Im_Psi.flatten(),
    'Gauge_Lambda_Z': Gauge_Z.flatten(),
    'Heat_Probability_C': Heat.flatten()
})

df.to_csv('Hilbert_Thimble_Surface.csv', index=False)
print("Hilbert Surface generated! Follow the OriginLab Scatter instructions.")