import numpy as np
import pandas as pd

# --- 1. THE ABRUPT PHYSICS PARAMETERS ---
k_pani = 8.0      # Frequency of PAni wave
k_tio2 = 5.0      # Frequency of TiO2 wave
T = 0.3           # Transmission Coefficient (The instantaneous "decay" penalty caused by the Dirac Delta at x=0)

E_total = 1.0     
wave_scale = 0.5  

# --- 2. DEFINE THE LIMIT-BASED DOMAINS (0- and 0+) ---
# Notice there is no gap. They touch perfectly at exactly x=0.
x_pani = np.linspace(-4, 0, 400, endpoint=True)   # x -> 0-
x_tio2 = np.linspace(0, 4, 400, endpoint=True)    # x -> 0+

# --- 3. CALCULATE THE ABRUPT WAVEFUNCTION ---
psi_pani = np.cos(k_pani * x_pani)

# The wave instantly emerges with reduced amplitude T
psi_tio2 = T * np.cos(k_tio2 * x_tio2)

# --- 4. ASSEMBLE THE DATA ---
x_total = np.concatenate((x_pani, x_tio2))
psi_raw = np.concatenate((psi_pani, psi_tio2))
prob_density = np.abs(psi_raw)**2

psi_shifted = E_total + (psi_raw * wave_scale)
prob_shifted = E_total + (prob_density * wave_scale)

# --- 5. THE ABRUPT POTENTIAL STEP ---
v_total = np.zeros_like(x_total)
# A single vertical singularity at x=0
v_total[x_total == 0] = 2.5 

df = pd.DataFrame({
    'Position_x': x_total,
    'Dirac_Potential_V': v_total,
    'Total_Energy_E': np.full_like(x_total, E_total),
    'Wavefunction_Shifted': psi_shifted,
    'Probability_Density': prob_shifted
})

df.to_csv('Abrupt_Dirac_Heterojunction.csv', index=False)
print("True Abrupt Heterojunction generated. No spatial decay, only instant amplitude shift at x=0.")