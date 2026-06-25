import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# --- 1. AXIOMATIC SYSTEM PARAMETERS ---
hbar = 1.0
x_min, x_max = -50.0, 50.0   
N_x = 2000               
dx = (x_max - x_min) / (N_x - 1)
x = np.linspace(x_min, x_max, N_x)

t_total = 15.0           
N_t = 600                
dt = t_total / N_t
time_array = np.linspace(0, t_total, N_t)

# --- 2. PHASE-DEPENDENT EFFECTIVE MASS TENSOR ---
# m_PAni = 1.0 (Baseline Organic Polymer)
# m_TiO2 = 0.5 (Anatase: Light, Fast, Low DOS/Ip)
# m_TiO2 = 5.0 (Rutile: Heavy, Slow/Delayed, High DOS/Ip)
m_PAni = 1.0
m_TiO2 = 0.5  # Set to > 1.0 to simulate the Rutile delay effect

m_eff = np.full(N_x, m_PAni)
m_eff[x >= 0] = m_TiO2

# Calculate interpolated mass boundaries for symmetrized operator
m_mid = 0.5 * (m_eff[:-1] + m_eff[1:])

# --- 3. ENERGY LANDSCAPE ---
V0 = 1.5                 
V = np.zeros(N_x)
V[x < 0] = V0            

# --- 4. INITIALIZATION (PAni Polaron) ---
x0 = -15.0               
sigma = 2.0              
k0 = 2.5                 
A = (1.0 / (np.pi * sigma**2))**0.25
Psi = A * np.exp(-((x - x0)**2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
Psi = Psi / np.sqrt(np.sum(np.abs(Psi)**2 * dx))

# --- 5. HERMITIAN HAMILTONIAN CONSTRUCTION (von Roos) ---
# Diagonal elements: hbar^2 / (2 * dx^2) * [1/m_{i-1/2} + 1/m_{i+1/2}] + V_i
diag_main = np.zeros(N_x)
diag_main[1:-1] = (hbar**2 / (2.0 * dx**2)) * (1.0 / m_mid[:-1] + 1.0 / m_mid[1:]) + V[1:-1]
diag_main[0] = diag_main[1]; diag_main[-1] = diag_main[-2] # Boundary handling

# Off-diagonal elements: -hbar^2 / (2 * dx^2) * [1/m_{i+1/2}]
diag_off = -(hbar**2 / (2.0 * dx**2)) * (1.0 / m_mid)

H = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')

# --- 6. CRANK-NICOLSON OPERATORS ---
I = sparse.eye(N_x, format='csc')
A_op = I + (0.5j * dt / hbar) * H
B_op = I - (0.5j * dt / hbar) * H
solve_A = spla.factorized(A_op)

# --- 7. TIME EVOLUTION & DATA COLLECTION ---
heatmap_matrix = np.zeros((N_t, N_x))

for n in range(N_t):
    heatmap_matrix[n, :] = np.abs(Psi)**2
    RHS = B_op.dot(Psi)
    Psi = solve_A(RHS)

# --- 8. EXACT TRANSMISSION & RECTIFIED DATA SERIALIZATION ---
final_prob_density = np.abs(Psi)**2
T_exact = np.sum(final_prob_density[x >= 0]) * dx

print(f"Simulation Complete. Phase Model: m_TiO2 = {m_TiO2}")
print(f"Exact Transmission Coefficient (T): {T_exact:.4f}")
print("Note: Correlate T_exact directly to the CV Peak Current (Ip).")

# =====================================================================
# MODIFIED EXPORT LOGIC FOR ORIGINPRO MATRIX COMPATIBILITY
# =====================================================================
# 1. Export exclusively the purely numerical probability amplitudes.
#    By setting header=False and index=False, we strip the spatial 
#    and temporal coordinates that corrupt the OriginPro Colormap.
df_heatmap = pd.DataFrame(heatmap_matrix)
df_heatmap.to_csv('Phase_Dependent_Mass_Heatmap_Raw.csv', index=False, header=False)

# 2. Serialize the spatiotemporal discretization arrays independently.
#    These can be imported into OriginPro separately to accurately 
#    calibrate the X and Y bounds of your Matrix image.
np.savetxt('spatial_coordinates_X.txt', x)
np.savetxt('temporal_coordinates_T.txt', time_array)
# =====================================================================