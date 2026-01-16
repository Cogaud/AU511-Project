# --- Main Execution ---
import numpy as np
from equilibrum import find_equilibrium_point52
from longitudinal_model import *
from analize import *
from sisopy31 import damp
from control import matlab

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
    sys_q_bode(A_m, B_m, C_m, D_m)
    # gain 
    Kr = -0.09
    Ak = A_m - Kr * B_m @ C_m[2, :].reshape(1,5)
    Bk = Kr * B_m
    Ck = C_m[2, :].reshape(1,5)
    Dk = Kr * D_m[2, :].reshape(1,1)
    Tqcl_ss, Tqcl_tf = closed_loop_q_qc(Ak, Bk, Ck, Dk)
    print("\n Closed Loop Poles & zeros q/q_c:")
    print(Tqcl_ss.poles())
    print(Tqcl_ss.zeros())

    # closed loop with filter
    tau = 0.05
    filter = control.tf([tau, 0], [tau, 1])
    tf_filter_out_a = control.series(control.feedback(Kr, control.series(filter, TqDm_tf_m)), TaDm_tf_m)

    # gamma feedback loop
    sys_gamma_bode(A_m, B_m, C_m, D_m)
    # gain
    Kg = -0.01 # to modify
    # Kg = -0.0001 # to modify

    Ag = Ak - Kg * Bk @ C_m[0, :].reshape(1,5)
    Bg = Kg * Bk
    Cg = C_m[0, :].reshape(1,5)
    Dg = Kr * D_m[0, :].reshape(1,1)
    Tgcl_ss, Tgcl_tf = closed_loop_g_gc_plot(Ag, Bg, Cg, Dg)
    print("\n Closed Loop Poles & zeros g/g_c:")
    print(Tgcl_ss.poles())
    print(Tgcl_ss.zeros())
    # add damp/ proper pulse

    # z feedback loop
    Kz = -0.0005 # to modify
    Az = Ag - Kz * Bg @ C_m[4, :].reshape(1,5)
    Bz = Kz * Bg
    Cz = C_m[4, :].reshape(1,5)
    Dz = Kz * Dg.reshape(1,1)
    Tzcl_ss, Tzcl_tf = closed_loop_z_zc_plot(Az, Bz, Cz, Dz)
    print("\n Closed Loop Poles & zeros z/z_c:")
    print(Tzcl_ss.poles())
    print(Tzcl_ss.zeros())
    # add damp/ proper pulse


   # addition of a saturation inn gamma control loop
    sys_sat = sys2
    print(sys_sat)
    delta_n_z_max = 3 # g
    # alpha = 6 # input angle of attack in deg
    # a_0 = alpha - (((2 * m * g * n_z) / (rho * S * V**2 * Cz_alpha)))
    # a_eq = (((2 * m * g * n_z) / (rho * S * V**2 * Cz_alpha))) + a_0
    # n_z = 1 + (alpha - a_eq)/(a_eq - a_0)
    # delta_n_z = (alpha - a_eq) / (a_eq - a_0)
    # a_max = a_eq + (a_eq - a_0) * delta_n_z




