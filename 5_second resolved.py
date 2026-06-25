import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# --- 1. AXIOMATIC SYSTEM PARAMETERS (Atomic / Scaled Units) ---
hbar = 1.0
m_eff = 1.0              # Effective mass 
x_min, x_max = -50, 50   # Spatial domain (R^1)
N_x = 2000               # Spatial grid resolution
dx = (x_max - x_min) / (N_x - 1)
x = np.linspace(x_min, x_max, N_x)

# Time evolution parameters
t_total = 15.0           # Total dimensionless time
N_t = 600                # Number of time steps
dt = t_total / N_t

# Wavepacket initial conditions (Polaron in PAni)
x0 = -15.0               # Initial position (left of heterojunction)
sigma = 2.0              # Spatial localization spread
k0 = 2.5                 # Initial incident wavevector (momentum towards x=0)

# --- 2. THE STEP-DOWN POTENTIAL LANDSCAPE ---
# LUMO of PAni is higher than CBM of TiO2. 
# Electron accelerates from V0 down to 0.0 at x=0.
V0 = 1.5                 
V = np.zeros(N_x)
V[x < 0] = V0            # PAni is the higher energy shelf
# V[x >= 0] remains 0.0 (TiO2 CBM baseline)

# --- 3. INITIALIZATION OF THE WAVEPACKET ---
# Gaussian envelope multiplied by the complex phase factor
A = (1.0 / (np.pi * sigma**2))**0.25
Psi_0 = A * np.exp(-((x - x0)**2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)

# Normalize strictly to ensure total probability = 1
Psi_0 = Psi_0 / np.sqrt(np.sum(np.abs(Psi_0)**2 * dx))

# --- 4. HAMILTONIAN CONSTRUCTION (Finite Difference) ---
# Kinetic energy operator: -hbar^2 / (2m) * d^2/dx^2
diag_main = (hbar**2) / (m_eff * dx**2) + V
diag_off = -(hbar**2) / (2.0 * m_eff * dx**2) * np.ones(N_x - 1)
H = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')

# --- 5. CRANK-NICOLSON OPERATORS ---
I = sparse.eye(N_x, format='csc')
A_op = I + (0.5j * dt / hbar) * H
B_op = I - (0.5j * dt / hbar) * H
solve_A = spla.factorized(A_op)

# --- 6. TIME EVOLUTION PROPAGATION ---
Psi = np.copy(Psi_0)
export_data = {'Position_x': x, 'Potential_V': V}
record_interval = max(1, N_t // 50) 

for n in range(N_t):
    # Propagate: Psi(t+dt) = A^-1 * B * Psi(t)
    RHS = B_op.dot(Psi)
    Psi = solve_A(RHS)
    
    if n % record_interval == 0:
        prob_density = np.abs(Psi)**2
        export_data[f'Time_{n*dt:.2f}'] = prob_density

# --- 7. EXACT TRANSMISSION COEFFICIENT CALCULATION ---
# Integrating the probability density strictly in the TiO2 domain (x >= 0)
# at the final time step (after the scattering event has resolved).
final_prob_density = np.abs(Psi)**2
T_exact = np.sum(final_prob_density[x >= 0]) * dx
R_exact = np.sum(final_prob_density[x < 0]) * dx

print(f"Simulation Complete.")
print("-" * 40)
print(f"Exact Transmission Coefficient (T): {T_exact:.4f} ({T_exact*100:.2f}%)")
print(f"Exact Reflection Coefficient (R):   {R_exact:.4f} ({R_exact*100:.2f}%)")
print(f"Total Probability (Unitarity):      {T_exact + R_exact:.4f}")
print("-" * 40)

# --- 8. EXPORT TO CSV FOR ORIGIN ---
df = pd.DataFrame(export_data)
df.to_csv('TDSE_StepDown_Scattering.csv', index=False)