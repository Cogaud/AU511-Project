import numpy as np
from math import sin, cos, pi, sqrt, atan, tan
# --- 3. Mode Analysis ---

def analyze_modes(A):
    eigenvalues = np.linalg.eigvals(A)
    print("\n--- Open Loop Modes Analysis ---")
    
    # Store modes for classification
    complex_modes = []
    real_modes = []
    
    # Helper to check if value is already processed (for conjugates)
    processed_indices = set()
    
    for i in range(len(eigenvalues)):
        if i in processed_indices:
            continue
            
        eig = eigenvalues[i]
        
        # Check for complex pair
        if abs(eig.imag) > 1e-6:
            # Find conjugate
            conj_found = False
            for j in range(i + 1, len(eigenvalues)):
                if j not in processed_indices:
                    other = eigenvalues[j]
                    if abs(other.real - eig.real) < 1e-5 and abs(other.imag + eig.imag) < 1e-5:
                        processed_indices.add(j)
                        conj_found = True
                        break
            
            omega_n = sqrt(eig.real ** 2 + eig.imag ** 2)
            damping = -eig.real / omega_n
            period = 2 * pi / abs(eig.imag)
            complex_modes.append({
                'eig': eig,
                'wn': omega_n,
                'zeta': damping,
                'period': period
            })
        else:
            real_modes.append(eig.real)

    # Sort complex modes by frequency (Low freq = Phugoid, High freq = Short Period)
    complex_modes.sort(key=lambda x: x['wn'])
    
    print(f"{'Mode Type':<15} | {'Eigenvalue':<28} | {'Freq (rad/s)':<12} | {'Damping':<10} | {'Period (s)':<10}")
    print("-" * 85)
    
    # Classify and Print Complex Modes
    for i, mode in enumerate(complex_modes):
        name = "Oscillatory"
        # Simple heuristic for longitudinal: 
        # If we have exactly 2 pairs, lower is Phugoid, higher is Short Period
        if len(complex_modes) == 2:
            if i == 0: name = "Phugoid"
            else: name = "Short Period"
            
        e = mode['eig']
        print(f"{name:<15} | {e.real:.4f} ± {abs(e.imag):.4f}j    | {mode['wn']:<12.4f} | {mode['zeta']:<10.4f} | {mode['period']:<10.2f}")

    # Print Real Modes
    for r in real_modes:
        print(f"{'Real':<15} | {r:.4f} + 0.0000j          | {'-':<12} | {'-':<10} | {'-':<10}")

    return eigenvalues


def calculate_short_period_approximation(A):
    """
    Calculates Short Period mode characteristics using the reduced order approximation.
    Assumes state vector: [V, gamma, alpha, q, theta, z]
    Uses submatrix for alpha (idx 2) and q (idx 3).
    """
    print("\n--- Short Period Approximation (Reduced Order) ---")
    
    # Extract submatrix for alpha and q
    # Indices: 2 (alpha), 3 (q)
    A_sp = A[2:4, 2:4]
    
    eigenvalues = np.linalg.eigvals(A_sp)
    
    # Check for complex pair
    # We expect a conjugate pair for standard stable short period
    
    # Check if complex
    if np.iscomplex(eigenvalues).any():
        # Take the one with positive imaginary part or just the first one
        eig = eigenvalues[0]
        
        # Ensure we work with the complex values
        wn = sqrt(eig.real**2 + eig.imag**2)
        zeta = -eig.real / wn
        period = 2 * pi / abs(eig.imag) if eig.imag != 0 else float('inf')
        
        print(f"  Eigenvalues:       {eig.real:.4f} ± {abs(eig.imag):.4f}j")
        print(f"  Natural Freq (wn): {wn:.4f} rad/s")
        print(f"  Damping (zeta):    {zeta:.4f}")
        print(f"  Period:            {period:.4f} s")
        
        return wn, zeta
    else:
        print("  Approximation yielded real eigenvalues (Non-oscillatory).")
        print(f"  Eigenvalues: {eigenvalues.real}")
        return None

def calculate_phugoid_approximation(A):
    """
    Calculates Phugoid mode characteristics using the reduced order approximation.
    Assumes state vector: [V, gamma, alpha, q, theta, z]
    Uses submatrix for alpha (idx 2) and q (idx 3).
    """
    print("\n--- Phugoid Approximation (Reduced Order) ---")
    
    # Extract submatrix for alpha and q
    # Indices: 2 (alpha), 3 (q)
    A_sp = A[0:2, 0:2]
    
    eigenvalues = np.linalg.eigvals(A_sp)
    
    # Check for complex pair
    # We expect a conjugate pair for standard stable short period
    
    # Check if complex
    if np.iscomplex(eigenvalues).any():
        # Take the one with positive imaginary part or just the first one
        eig = eigenvalues[0]
        
        # Ensure we work with the complex values
        wn = sqrt(eig.real**2 + eig.imag**2)
        zeta = -eig.real / wn
        period = 2 * pi / abs(eig.imag) if eig.imag != 0 else float('inf')
        
        print(f"  Eigenvalues:       {eig.real:.4f} ± {abs(eig.imag):.4f}j")
        print(f"  Natural Freq (wn): {wn:.4f} rad/s")
        print(f"  Damping (zeta):    {zeta:.4f}")
        print(f"  Period:            {period:.4f} s")
        
        return wn, zeta
    else:
        print("  Approximation yielded real eigenvalues (Non-oscillatory).")
        print(f"  Eigenvalues: {eigenvalues.real}")
        return None

from scipy import signal

def transfert_function(A, B, C, D):
    """
    Calculates the transfer function of the system.
    """
    # Create the state space system
    sys = signal.StateSpace(A, B, C, D)
    
    # Convert to Transfer Function representation
    # sys.to_tf() returns (num, den)
    # However, since we have MIMO (1 input, 6 outputs), 
    # it returns num as (outputs, inputs, degree+1) and den as (outputs, degree+1)
    # But usually den is common for all outputs in a minimal realization.
    # scipy ss2tf returns:
    # num : 2-D ndarray
    # den : 1-D ndarray or 2-D ndarray
    
    # Let's use ss2tf directly to be safe with dimensions
    # ss2tf(A, B, C, D, input=0) -> for the single input we have
    num, den = signal.ss2tf(A, B, C, D, input=0)
    
    # Format the output string
    output_str = ""
    state_names = ["V", "gamma", "alpha", "q", "theta", "z"]
    
    for i in range(len(output_str), len(num)):
        if i < len(state_names):
            name = state_names[i]
        else:
            name = f"State {i+1}"
            
        n = num[i]
        d = den
        
        # Create polynomial strings
        # We can use np.poly1d to help format, or build manually
        n_poly = np.poly1d(n)
        d_poly = np.poly1d(d)
        
        output_str += f"\n--- Transfer Function for {name} / delta_m ---\n"
        output_str += f"{n_poly}\n"
        output_str += "-" * 50 + "\n"
        output_str += f"{d_poly}\n"
        
    return output_str