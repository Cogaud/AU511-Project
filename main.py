# --- Main Execution ---
import numpy as np
from equilibrum import find_equilibrium_point52
from longitudinal_model import *
from analize import *
from sisopy31 import damp
# from control import matlab
import control
import matplotlib.pyplot

if __name__ == "__main__":
    # Operating Point
    Mach_op = 0.95
    Alt_op = 11812   # ft

    # 1. Find Equilibrium
    eq_res = find_equilibrium_point52(Mach_op, Alt_op)

    print(f"\nEquilibrium Results:")
    print(f"  Velocity: {eq_res['V_eq']:.2f} m/s")
    print(f"  Alpha:    {eq_res['alpha_eq_deg']:.3f} deg")
    print(f"  Delta_m:  {eq_res['delta_m_eq_deg']:.3f} deg")
    print(f"  Thrust:   {eq_res['Thrust']:.2f} N")
    print(f"  Cz:       {eq_res['Cz_eq']:.4f}")
    print(f"  Cx:       {eq_res['Cx_eq']:.4f}")

    # 2. State Space
    A, B, C, D = get_state_space(eq_res)

    print("\nState Space Matrix A:")
    print(np.array2string(A, precision=4, suppress_small=True))

    print("\nState Space Matrix B:")
    print(np.array2string(B, precision=4, suppress_small=True))

    # 3. Analyze Modes
    analyze_modes(A)

    # 4. Short period analysis
    calculate_short_period_approximation(A)
    short_period_linearization(A, B, C, D)

    # 5. Phugoid analysis
    calculate_phugoid_approximation(A)
    phugoid_linearization(A, B, C, D)

# --- New System Definition ---
# We removed the first row and first column corresponding to V (forward velocity)
    print("\n New System (without V):")
    A_m = A[1:6,1:6]
    B_m = B[1:6,0:1]
    C_m = np.eye(5)
    D_m = np.zeros((5, 1))
    sys_m = control.ss(A_m, B_m, C_m, D_m)
    print("\n Transfer Function of the new system:")
    tf_m = control.ss2tf(sys_m)
    print(tf_m)
    TgDm_tf_m = control.ss2tf(control.ss(A_m, B_m, C_m[0, :], D_m[0, :]))
    TaDm_tf_m = control.ss2tf(control.ss(A_m, B_m, C_m[1, :], D_m[1, :]))
    TqDm_tf_m = control.ss2tf(control.ss(A_m, B_m, C_m[2, :], D_m[2, :]))
    TtDm_tf_m = control.ss2tf(control.ss(A_m, B_m, C_m[3, :], D_m[3, :]))
    TzDm_tf_m = control.ss2tf(control.ss(A_m, B_m, C_m[4, :], D_m[4, :]))
    print("\n Poles and Zeros of the new system:")
    print(sys_m.poles())
    print(sys_m.zeros())

# 8. closed loop q/qc
    # sys_q_bode(A_m, B_m, C_m, D_m)
    # gain 
    Kr = -0.09
    Ak = A_m - Kr * B_m * C_m[2, : ]
    Bk = Kr * B_m
    Ck = C_m[2, : ]
    Dk = D_m[2, : ]
    sys_q_dm = control.ss(Ak, Bk, C_m, D_m)
    sys_q_qc = control.ss(Ak, Bk, Ck, Dk)
    sys_q_tf = control.ss2tf(sys_q_qc)

    print("\n Closed loop Transfer Function q/q_c:")
    print(sys_q_tf)
    print("\n Closed Loop Poles & zeros q/q_c:")
    poles_q = control.poles(sys_q_tf)
    print(poles_q)
    zeros_q = control.zeros(sys_q_tf)
    print(zeros_q)
    print("\n Damping ratios and natural frequencies of closed loop q/qc:")
    damp_q = damp(sys_q_tf)
    print("Damping ratio : ", damp_q[3])
    print("Natural frequency (rad/s) : ", damp_q[4])
    print("\n Step Response of closed loop q/qc:")
    Yqcl, Tqcl = control.matlab.step(sys_q_tf, arange(0, 5, 0.01))
    plot(Tqcl, Yqcl, 'b', lw=2)
    plot([0, Tqcl[-1]], [Yqcl[-1], Yqcl[-1]], 'k--', lw=1)
    plot([0, Tqcl[-1]], [1.05*Yqcl[-1], 1.05*Yqcl[-1]], 'k--', lw=1)
    plot([0, Tqcl[-1]], [0.95*Yqcl[-1], 0.95*Yqcl[-1]], 'k--', lw=1)
    minorticks_on()
    # grid(b=True, which='both')
    grid(True)
    title('Step Response $q/q_c$')
    xlabel('Time (s)')
    ylabel(r'$q$ (rad/s)')
    show()

    # closed loop with filter
    tau = 0.05  # to modify
    filter = control.tf([tau, 0], [tau, 1])
    tf_filter_out_a = control.series(control.feedback(Kr, control.series(TqDm_tf_m, filter)), TaDm_tf_m)


# gamma feedback loop
    # sys_gamma_bode(Ak, Bk, C_m, D_m)
    # gain
    Kg = 11 # to modify
    # Kg = -0.0001 # to modify

    Ag = Ak - Kg * Bk * C_m[0, : ]
    Bg = Kg * Bk
    Cg = C_m[0, : ]
    Dg = D_m[0, : ]
    sys_g_dm = control.ss(Ag, Bg, C_m, D_m)
    sys_g_gc = control.ss(Ag, Bg, Cg, Dg)
    sys_g_tf = control.ss2tf(sys_g_gc)

    print("\n Closed loop Transfer Function g/g_c:")
    print(sys_g_tf)
    print("\n Closed Loop Poles & zeros g/g_c:")
    poles_g = control.poles(sys_g_tf)
    print(poles_g)
    zeros_g = control.zeros(sys_g_tf)
    print(zeros_g)
    print("\n Damping ratios and natural frequencies of closed loop g/g_c:")
    damp_g = damp(sys_g_tf)
    print("Damping ratio : ", damp_g[3])
    print("Natural frequency (rad/s) : ", damp_g[4])
    print("\n Step Response of closed loop g/g_c:")
    Ygcl, Tgcl = control.matlab.step(sys_g_tf, arange(0, 5, 0.01))
    plot(Tgcl, Ygcl, 'b', lw=2)
    plot([0, Tgcl[-1]], [Ygcl[-1], Ygcl[-1]], 'k--', lw=1)
    plot([0, Tgcl[-1]], [1.05*Ygcl[-1], 1.05*Ygcl[-1]], 'k--', lw=1)
    plot([0, Tgcl[-1]], [0.95*Ygcl[-1], 0.95*Ygcl[-1]], 'k--', lw=1)
    minorticks_on()
    # grid(b=True, which='both')
    grid(True)
    title('Step Response $g/g_c$')
    xlabel('Time (s)')
    ylabel(r'$g$ (rad/s)')
    show()

    # z feedback loop
    # sys_z_bode(Ag, Bg, C_m, D_m)
    Kz = 0.00379 # to modify

    Az = Ag - Kz * Bg * C_m[4, : ]
    Bz = Kz * Bg
    Cz = C_m[4, : ]
    Dz = D_m[4, : ]
    sys_z_dm = control.ss(Az, Bz, C_m, D_m)
    sys_z_zc = control.ss(Az, Bz, Cz, Dz)
    sys_z_tf = control.ss2tf(sys_z_zc)
    print("\n Closed loop Transfer Function z/z_c:")
    print(sys_z_tf)
    print("\n Closed Loop Poles & zeros z/z_c:")
    poles_z = control.poles(sys_z_tf)
    print(poles_z)
    zeros_z = control.zeros(sys_z_tf)
    print(zeros_z)
    print("\n Damping ratios and natural frequencies of closed loop z/z_c:")
    damp_z = damp(sys_z_tf)
    print("Damping ratio : ", damp_z[3])
    print("Natural frequency (rad/s) : ", damp_z[4])
    print("\n Step Response of closed loop z/z_c:")
    Yzcl, Tzcl = control.matlab.step(sys_z_tf, arange(0, 5, 0.01))
    plot(Tzcl, Yzcl, 'b', lw=2)
    plot([0, Tzcl[-1]], [Yzcl[-1], Yzcl[-1]], 'k--', lw=1)
    plot([0, Tzcl[-1]], [1.05*Yzcl[-1], 1.05*Yzcl[-1]], 'k--', lw=1)
    plot([0, Tzcl[-1]], [0.95*Yzcl[-1], 0.95*Yzcl[-1]], 'k--', lw=1)
    minorticks_on()
    # grid(b=True, which='both')
    grid(True)
    title('Step Response $z/z_c$')
    xlabel('Time (s)')
    ylabel(r'$z$ (rad/s)')
    show()


    # --- Saturation Analysis: Find γ_max ---
    print("\n" + "="*60)
    print("SATURATION ANALYSIS: Finding γ_max for Δn_z = 3g")
    print("="*60)

    # Créer le système complet en boucle fermée (γ et q boucles)
    # avec toutes les 5 sorties (γ, α, q, θ, z)
    sys_closed = sys_g_dm
    sys_cl_tf = control.ss2tf(sys_closed)

    # Paramètres
    delta_n_z = 3.0  # 3g load factor
    gamma_min = -1.0
    gamma_max = 1.0

    # Trouver γ_max
    try:
        gamma_optimal = find_gamma_max(
            sys_cl_tf, 
            delta_n_z=delta_n_z, 
            eq_res=eq_res, 
            gamma_min=gamma_min, 
            gamma_max=gamma_max
        )
        
        print(f"\nOptimal flight path angle: γ_max = {gamma_optimal:.4f} rad = {np.degrees(gamma_optimal):.2f}°")
        
        # Vérification : calculer alpha_max
        alpha_max = saturation_analysis(eq_res, delta_n_z)
        print(f"Maximum angle of attack: α_max = {alpha_max:.4f} rad = {np.degrees(alpha_max):.2f}°")
        
        # Afficher la réponse pour ce γ_max
        response = response_alpha_to_gamma(sys_cl_tf, gamma_optimal, delta_n_z, eq_res)
        print(f"Verification: max(α(t)) - α_max = {response:.6f} (should be ≈ 0)")
        
    except Exception as e:
        print(f"Error in saturation analysis: {e}")




