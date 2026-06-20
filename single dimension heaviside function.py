import numpy as np
import pandas as pd

# parameters
A = 1.0       # amplitude peak for wavefunction
kappa = 2.0   # decay constant for evanescent
k = 5.0       # wavevector for bloch TiO2
W = 1.5       # coupling strength (delta V bridge) *subjected
hbar = 1.0    # normalized planck

# spatial data (wavefunction vs pos)
x_pani = np.linspace(-3, 0, 500)      # X-coordinates for PAni (x < 0)
x_tio2 = np.linspace(0, 3, 500)       # X-coordinates for TiO2 (x > 0)

psi_pani = A * np.exp(kappa * x_pani) # exp decay
psi_tio2 = A * np.cos(k * x_tio2)     # bloch wave

# combination
x_total = np.concatenate((x_pani, x_tio2))
psi_total = np.concatenate((psi_pani, psi_tio2))

# temp data for P vs t 
time = np.linspace(0, 5, 1000)
p_tunnel = np.cos(W * time / hbar)**2
p_transfer = np.sin(W * time / hbar)**2

# export 
spatial_df = pd.DataFrame({'Position (x)': x_total, 'Wave Amplitude (Psi)': psi_total})
temporal_df = pd.DataFrame({'Time (t)': time, 'P_PAni': p_tunnel, 'P_TiO2': p_transfer})

spatial_df.to_csv('Spatial_Wavefunction_Origin.csv', index=False)
temporal_df.to_csv('Temporal_Probability_Origin.csv', index=False)