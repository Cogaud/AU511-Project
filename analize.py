from __future__ import unicode_literals

import numpy as np
from math import sin, cos, pi, sqrt, atan, tan
import matplotlib.pyplot as plt
import control
import sisopy31 as siso
from scipy.optimize import brentq
from aircraft_data import *
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
    Uses submatrix for V (idx 0) and gamma (idx 1).
    """
    print("\n--- Phugoid Approximation (Reduced Order) ---")
    
    # Extract submatrix for alpha and q
    # Indices: 0 (V), 1 (gamma)
    A_sp = A[0:2, 0:2]
    
    eigenvalues = np.linalg.eigvals(A_sp)
    
    # Check for complex pair
    # We expect a conjugate pair for standard stable phugoid
    
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

def transfert_function(A, B, C, D):
    """
    Calculates the transfer function of the system.
    """
    # Create the state space system using control library
    sys = control.ss(A, B, C, D)
    
    # Convert to Transfer Function representation
    # control.ss2tf returns a TransferFunction object which has a nice string representation
    tf = control.ss2tf(sys)
    
    return tf

def pole_info(A): 
    eigenvalues = np.linalg.eigvals(A)

    # Check for complex pair
    # We expect a conjugate pair for standard stable phugoid
    
    # Check if complex
    if np.iscomplex(eigenvalues).any():
        # Take the one with positive imaginary part or just the first one
        eig = eigenvalues[0]
        
        # Ensure we work with the complex values
        wn = sqrt(eig.real**2 + eig.imag**2)
        zeta = -eig.real / wn
        period = 2 * pi / abs(eig.imag) if eig.imag != 0 else float('inf')
        
        print(f"  Natural Freq (wn): {wn:.4f} rad/s")
        print(f"  Damping (zeta):    {zeta:.4f}")
        print(f"  Period:            {period:.4f} s")

def ploting_step_response(A, B, C, D):
    """
    Plots the step response of the system.
    Automatically determines simulation time based on Phugoid period.
    """
    # Create the state space system
    sys = control.ss(A, B, C, D)
    
    # 1. Determine suitable time vector based on Phugoid Mode
    vals = np.linalg.eigvals(A)
    # Filter for complex modes with low frequency
    complex_modes = [e for e in vals if abs(e.imag) > 1e-6]
    complex_modes.sort(key=lambda x: abs(x)) # Sort by magnitude (freq approx)
    
    # Default time
    T_final = 100
    
    # Try to find Phugoid (lowest freq complex mode)
    if complex_modes:
        # Assuming Phugoid is the lowest frequency oscillatory mode
        phugoid_eig = complex_modes[0] 
        wn_ph = abs(phugoid_eig)
        # Period = 2*pi / damped_freq approx 2*pi/wn for low damping
        # But correctly: 2*pi / imag
        period_ph = 2 * pi / abs(phugoid_eig.imag)
        
        print(f"  Detected Phugoid Period for Plotting: {period_ph:.2f} s")
        T_final = 5 * period_ph
    
    # Create custom time vector
    T = np.linspace(0, T_final, 2000)
    
    # Compute step response
    # T: time vector
    # yout: response (outputs, inputs, time) or (outputs, time)
    res = control.step_response(sys, T)
    yout = res.outputs
    
    # Check dimensions
    # If 3D array (outputs, inputs, time), squeeze input dimension
    if yout.ndim == 3:
        yout = yout[:, 0, :]
        
    # Plot Phugoid Variables: V (idx 0), gamma (idx 1)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.suptitle('Phugoid Mode Response (Long Term)')
    
    # V / delta_m (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Amplitude', color=color)
    ax1.plot(T, yout[0], color=color, linewidth=2, label=r'$V / \delta_m$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # gamma / delta_m (Right Axis)
    color = 'tab:red'
    ax1.set_ylabel('Amplitude', color=color)  # we already handled the x-label with ax1
    ax1.plot(T, yout[1], color=color, linewidth=2, label=r'$\gamma / \delta_m$')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    
    plt.tight_layout()

    # Plot Short Period Variables: alpha (idx 2), q (idx 3)
    # Short period is fast, so we might want a shorter timescale for this,
    # but the user specifically asked for Phugoid fixes. I'll stick to the same T for now
    # or zoom in. Let's make a separate figure with shorter time if needed, 
    # but typically one looks at the first few seconds for Short Period.
    
    # Let's calculate Short Period T separately
    T_final_sp = T_final / 20 # Short period is much faster
    if len(complex_modes) >= 4: # If we have 2 pairs
         sp_eig = complex_modes[-1] # Highest freq
         period_sp = 2 * pi / abs(sp_eig.imag)
         T_final_sp = 10 * period_sp # 10 periods
    
    T_sp = np.linspace(0, T_final_sp, 1000)
    res_sp = control.step_response(sys, T_sp)
    yout_sp = res_sp.outputs
    if yout_sp.ndim == 3:
        yout_sp = yout_sp[:, 0, :]

    fig2, ax3 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Short Period Mode Response (Short Term)')
    
    # alpha / delta_m (Left Axis)
    color = 'tab:green'
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Amplitude', color=color)
    ax3.plot(T_sp, yout_sp[2], color=color, linewidth=2, label=r'$\alpha / \delta_m$')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True)
    
    # q / delta_m (Right Axis)
    color = 'k'
    ax3.set_ylabel('Amplitude', color=color)
    ax3.plot(T_sp, yout_sp[3], color=color, linewidth=2, label=r'$q / \delta_m$')
    ax3.tick_params(axis='y', labelcolor=color)
    
    # Combined legend
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines3, labels3, loc='best')
    
    plt.tight_layout()
    plt.show()

def sys_q_bode(A, B, C, D):
    """
    Plots the Bode plot for the pitch rate (q) response to elevator deflection (delta_m).
    Assumes state vector: [gamma, alpha, q, theta, z]
    Output: q (idx 2)
    Input: delta_m (assumed to be the first input, idx 0)
    """
    # Create the state space system
    Ai = A[1:3, 1:3]
    Bi = B[1:3, 0:1]
    Ciq = np.matrix ( [ [  0, 1 ] ] )
    Di = np.matrix ( [ [ 0 ] ] )
    TqDm_ss= control.ss ( Ai , Bi , Ciq , Di )
    TqDm_tf = control.ss2tf(TqDm_ss)
    siso.sisotool(TqDm_tf)

def sys_gamma_bode(A,B,C,D):
    """
    Plots the Bode plot for the angle (gamma) response to elevator deflection (delta_m).
    Assumes state vector: [V, gamma, alpha, q, theta, z]
    Output: gamma (idx 1)
    Input: delta_m (assumed to be the first input, idx 0)
    """
    # Create the state space system
    Cig = C[0, :]
    Di = D[0, :]
    TgDm_ss= control.ss ( A , B , Cig , Di )
    TgDm_tf = control.ss2tf(TgDm_ss)

    siso.sisotool(TgDm_tf)

def sys_z_bode(A,B,C,D):
    """
    Plots the Bode plot for the altitude z response to elevator deflection (delta_m).
    Assumes state vector: [gamma, alpha, q, theta, z]
    Output: z (idx 4)
    Input: delta_m (assumed to be the first input, idx 0)
    """
    Cz = np.matrix ( [ [ 0, 0, 0, 0, 1 ] ] )
    Di = np.matrix ( [ [ 0 ] ] )
    TzDm_ss= control.ss ( A , B , Cz , Di )
    TzDm_tf = control.ss2tf(TzDm_ss)

    siso.sisotool(TzDm_tf)

def saturation_analysis(eq_res, delta_n_z):
    """
    Analyze the effect of actuator saturation on the maximum angle of attack.
    """
    a_eq = eq_res['alpha_eq_deg']  # deg
    g = 9.81  # m/s2
    C_z_a = coefs['C_z_a']  # per radian
    rho = eq_res['rho']  # kg/m3
    V = eq_res['V_eq']  # m/s

    a_0 = a_eq - (((2 * m * g) / (rho * S * V**2 * C_z_a)))
    a_max = a_eq + (a_eq - a_0) * delta_n_z

    return a_max


