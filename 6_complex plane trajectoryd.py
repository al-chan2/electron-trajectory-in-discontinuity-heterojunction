import numpy as np
import pandas as pd

# --- 1. AXIOMATIC PARAMETERS ---
m_eff = 1.0
hbar = 1.0

# Energy Landscape (Step-down from PAni to TiO2)
E_total = 2.5            # Total energy of the electron (eV)
V_PAni = 1.5             # LUMO level of PAni (eV)
V_TiO2 = 0.0             # CBM level of TiO2 (eV)

# --- 2. WAVEVECTOR DERIVATION ---
# k = sqrt(2m(E-V))/hbar
k_PAni = np.sqrt(2.0 * m_eff * (E_total - V_PAni)) / hbar
k_TiO2 = np.sqrt(2.0 * m_eff * (E_total - V_TiO2)) / hbar

# --- 3. BOUNDARY CONDITION MATCHING (x=0) ---
# 1 + r = t
# k_PAni * (1 - r) = k_TiO2 * t
t = (2.0 * k_PAni) / (k_PAni + k_TiO2)
r = t - 1.0

# Calculate exact Transmission Probability for verification
T_exact = (k_TiO2 / k_PAni) * (np.abs(t)**2)

# --- 4. SPATIAL DOMAIN MAPPING ---
# We map the space near the singularity x=0
x_PAni = np.linspace(-10, 0, 500, endpoint=True)
x_TiO2 = np.linspace(0, 10, 500, endpoint=True)

# --- 5. WAVEFUNCTION ASSEMBLY ---
# PAni Domain: Incident + Reflected wave (Interference creates the Evanescent-like standing wave envelope)
Psi_PAni = np.exp(1j * k_PAni * x_PAni) + r * np.exp(-1j * k_PAni * x_PAni)

# TiO2 Domain: Transmitted Bloch wave
Psi_TiO2 = t * np.exp(1j * k_TiO2 * x_TiO2)

# Concatenate arrays
x_total = np.concatenate((x_PAni, x_TiO2))
Psi_total = np.concatenate((Psi_PAni, Psi_TiO2))

# --- 6. EXPORT HILBERT SPACE MAPPING FOR ORIGIN ---
df = pd.DataFrame({
    'Spatial_x': x_total,
    'Real_Psi': np.real(Psi_total),
    'Imag_Psi': np.imag(Psi_total),
    'Prob_Density': np.abs(Psi_total)**2
})

df.to_csv('Complex_Plane_Graph9.csv', index=False)

print(f"Mathematical Resolution Complete.")
print(f"Wavevector PAni (k1): {k_PAni:.4f}")
print(f"Wavevector TiO2 (k2): {k_TiO2:.4f}")
print(f"Transmission Amplitude (t): {t:.4f}  <-- This is your final radius in Graph 9")
print(f"Transmission Probability (T): {T_exact:.4f}")
print("\nInstructions for Origin (Graph 9):")
print("1. Plot 'Real_Psi' on the X-axis and 'Imag_Psi' on the Y-axis as a Line plot.")
print("2. The trajectory for x < 0 will show elliptical interference patterns.")
print("3. The trajectory for x > 0 will lock into a perfect circle of radius 't'.")