import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# --- 1. AXIOMATIC SYSTEM PARAMETERS ---
hbar = 1.0
m_eff = 1.0              
x_min, x_max = -40.0, 40.0   
N_x = 1000               
dx = (x_max - x_min) / (N_x - 1)
x = np.linspace(x_min, x_max, N_x)

t_total = 12.0           
N_t = 600                
dt = t_total / N_t
time_array = np.linspace(0, t_total, N_t)

# Polaron Initial Conditions
x0 = -15.0               
sigma = 2.0              
k0 = 2.5                 

# Step-Down Potential (PAni LUMO to TiO2 CBM)
V0 = 1.5                 
V = np.zeros(N_x)
V[x < 0] = V0            

# --- 2. INITIALIZATION ---
A = (1.0 / (np.pi * sigma**2))**0.25
Psi = A * np.exp(-((x - x0)**2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
Psi = Psi / np.sqrt(np.sum(np.abs(Psi)**2 * dx))

# --- 3. HAMILTONIAN & CRANK-NICOLSON OPERATORS ---
diag_main = (hbar**2) / (m_eff * dx**2) + V
diag_off = -(hbar**2) / (2.0 * m_eff * dx**2) * np.ones(N_x - 1)
H = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')

I = sparse.eye(N_x, format='csc')
A_op = I + (0.5j * dt / hbar) * H
B_op = I - (0.5j * dt / hbar) * H
solve_A = spla.factorized(A_op)

# --- 4. TIME EVOLUTION & DATA COLLECTION ---
# Initialize a 2D matrix to store probability densities for the heatmap
heatmap_matrix = np.zeros((N_t, N_x))

for n in range(N_t):
    heatmap_matrix[n, :] = np.abs(Psi)**2
    # Propagate to next time step
    RHS = B_op.dot(Psi)
    Psi = solve_A(RHS)

# --- 5. EXPORT FOR ORIGIN ---
# To plot a heatmap in Origin, a matrix format is optimal.
# Rows will be time steps, Columns will be spatial coordinates.
df_heatmap = pd.DataFrame(heatmap_matrix, columns=np.round(x, 2))
df_heatmap.insert(0, 'Time', np.round(time_array, 3))

csv_filename = 'Corrected_Graph2_Heatmap.csv'
df_heatmap.to_csv(csv_filename, index=False)

print(f"Simulation Complete. Data exported to '{csv_filename}'.")
print("\nInstructions for OriginLab:")
print("1. Import the CSV.")
print("2. Set the 'Time' column as the Y-coordinate. The column headers are the X-coordinates.")
print("3. Convert the worksheet to a Virtual Matrix (or regular Matrix).")
print("4. Plot as an Image Plot, Heatmap, or Contour - Color Fill.")

# --- 6. QUICK VISUALIZATION (Matplotlib) ---
plt.figure(figsize=(10, 6))
plt.imshow(heatmap_matrix, extent=[x_min, x_max, t_total, 0], 
           aspect='auto', cmap='hot', origin='upper')
plt.colorbar(label='Probability Density $|\Psi|^2$')
plt.axvline(x=0, color='white', linestyle='--', alpha=0.5, label='Heterojunction ($x=0$)')
plt.xlabel('Trajectory in $\mathbb{R}^1$')
plt.ylabel('Time (t)')
plt.title('Quantum Wavepacket Scattering Heatmap')
plt.legend()
plt.tight_layout()
plt.show()