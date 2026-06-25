import numpy as np
import pandas as pd

# --- 1. THE GAUGE PARAMETERS ---
# We use lambda (L) as the dimensionless transformation gauge (0 to 1)
resolution = 500
L = np.linspace(0, 1, resolution)  

# --- 2. THE SINGULARITY PHYSICS ---
A_pani = 1.0          # Initial Amplitude at x = 0- (lambda = 0)
T = 0.3               # Final Transmitted Amplitude at x = 0+ (lambda = 1)
phase_shift = np.pi   # The Dirac delta induces a phase rotation during the jump

# --- 3. THE MATHEMATICAL TRANSFORMATION ---
# The Amplitude mathematically crushes exponentially across the gauge
Amplitude = A_pani * np.exp(np.log(T) * L)

# The Phase rotates across the complex plane during the transformation
Phase = phase_shift * L

# --- 4. THE COMPLEX PLANE PROJECTION ---
# Euler's Formula: Psi = A * e^(i * Phase) = A * cos(Phase) + i * A * sin(Phase)
Re_Psi = Amplitude * np.cos(Phase)
Im_Psi = Amplitude * np.sin(Phase)

# --- 5. EXPORT TO ORIGINLAB ---
df = pd.DataFrame({
    'Gauge_Condition_Lambda': L,          # The progression from 0- to 0+
    'Real_Wavefunction': Re_Psi,          # X-axis in Origin
    'Imaginary_Wavefunction': Im_Psi,     # Y-axis in Origin
    'Absolute_Amplitude': Amplitude       # Optional Z-axis for 3D plotting
})

df.to_csv('Hilbert_Space_Singularity.csv', index=False)
print("Gauge Transformation generated! Import to OriginLab to view the Complex Plane trajectory.")