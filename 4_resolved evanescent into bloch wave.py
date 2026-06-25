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
k0 = 3.0                 # Initial incident wavevector (momentum towards x=0)

# Potential Energy Landscape (Heaviside Step at x=0)
V0 = 2.5                 # Conduction band offset / Barrier height
V = np.zeros(N_x)
V[x >= 0] = V0           # Theta(x) implementation

# --- 2. INITIALIZATION OF THE WAVEPACKET ---
# Gaussian envelope multiplied by the complex phase factor
A = (1.0 / (np.pi * sigma**2))**0.25
Psi_0 = A * np.exp(-((x - x0)**2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)

# Normalize strictly to ensure total probability = 1
Psi_0 = Psi_0 / np.sqrt(np.sum(np.abs(Psi_0)**2 * dx))

# --- 3. HAMILTONIAN CONSTRUCTION (Finite Difference) ---
# Kinetic energy operator: -hbar^2 / (2m) * d^2/dx^2
# Represented as a tridiagonal sparse matrix
diag_main = (hbar**2) / (m_eff * dx**2) + V
diag_off = -(hbar**2) / (2.0 * m_eff * dx**2) * np.ones(N_x - 1)

H = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')

# --- 4. CRANK-NICOLSON OPERATORS ---
I = sparse.eye(N_x, format='csc')
# Left hand side operator (A) and Right hand side operator (B)
A_op = I + (0.5j * dt / hbar) * H
B_op = I - (0.5j * dt / hbar) * H

# Pre-factorize A_op for highly efficient solving in the loop
solve_A = spla.factorized(A_op)

# --- 5. TIME EVOLUTION PROPAGATION ---
Psi = np.copy(Psi_0)

# Matrix to store data for Origin export
# Columns will be time steps, Rows will be spatial coordinates
export_data = {'Position_x': x, 'Potential_V': V}
record_interval = max(1, N_t // 50) # Save 50 slices for plotting

for n in range(N_t):
    # Propagate: Psi(t+dt) = A^-1 * B * Psi(t)
    RHS = B_op.dot(Psi)
    Psi = solve_A(RHS)
    
    # Store probability density |Psi|^2 periodically
    if n % record_interval == 0:
        prob_density = np.abs(Psi)**2
        export_data[f'Time_{n*dt:.2f}'] = prob_density

# --- 6. EXPORT TO CSV FOR ORIGIN ---
df = pd.DataFrame(export_data)
df.to_csv('TDSE_Wavepacket_Scattering.csv', index=False)
print("Simulation Complete. Data exported to 'TDSE_Wavepacket_Scattering.csv'.")
print("Instructions for Origin:")
print("1. Import CSV.")
print("2. Set 'Position_x' as X, 'Potential_V' as Y (optional, to overlay barrier).")
print("3. Select all 'Time_...' columns and plot as a 3D Waterfall, Surface, or Heatmap.")