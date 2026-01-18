# --- Main Execution ---
import numpy as np
from equilibrum import find_equilibrium_point52
from longitudinal_model import *
from analize import *
from sisopy31 import damp
# from control import matlab
import control
from scipy.optimize import bisect

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




# # --- Saturation Analysis: Find γ_max ---
# Créer le système complet en boucle fermée (γ et q boucles)
# avec toutes les 5 sorties (γ, α, q, θ, z)
sys_gamma_to_alpha = control.ss(Ag, Bg, C_m[1, :], D_m[1, :])
sys_gamma_to_alpha_tf = control.ss2tf(sys_gamma_to_alpha)

# Paramètres
delta_n_z = 3.0  # 3g load factor
gamma_min = -1.0
gamma_max = 1.0

alpha_max = saturation_analysis(eq_res, delta_n_z)

def f_gamma_sat(gamma_sat):
    t, alpha_response = control.step_response(sys_gamma_to_alpha_tf, T=np.linspace(0, 20, 2000))
    alpha_response_scaled = alpha_response * gamma_sat
    alpha_max_response = np.max(alpha_response_scaled)
    diff = alpha_max_response - alpha_max

    return diff

gamma_min = 0.001
gamma_max_search = 1.0 # rad
f_min = f_gamma_sat(gamma_min)
f_max = f_gamma_sat(gamma_max_search)
if f_min * f_max < 0:
    gamma_sat_solution = bisect(f_gamma_sat, gamma_min, gamma_max_search, xtol=1e-8)
    print("\nGamma_max = ", gamma_sat_solution)
else:
    print("\n Cannot apply bisection")
    for gmax in [0.5, 0.3, 0.2, 0.1]:
        f_test = f_gamma_sat(gmax)
        if f_min * f_test < 0:
            gamma_sat_solution = bisect(f_gamma_sat, gamma_min, gmax, xtol=1e-8)
            print("\n Gamma_max = ", gamma_sat_solution)
            break

# def find_gamma_max(sys_closed, delta_n_z=3.0, eq_res=None, gamma_min=-100, gamma_max=100):
#     """S
#     Trouve γ_max tel que max(α(t)) = delta_n_z (la cible)
#     Utilise la méthode de bisection (Brent's method).
#     """

#     def error_function(gamma_csat):
#         alpha_calc = response_alpha_to_gamma(sys_closed, gamma_csat, eq_res=eq_res)
#         return alpha_calc - delta_n_z
#     try:
#         gamma_optimal = brentq(error_function, gamma_min, gamma_max)
#     except ValueError as e:
#         print(f"Erreur d'optimisation : L'intervalle [{gamma_min}, {gamma_max}] ne contient pas la solution.")
#         print(f"Erreur aux bornes : f(min)={error_function(gamma_min):.2f}, f(max)={error_function(gamma_max):.2f}")
#         return None # Ou lever une erreur selon vos besoins

#     return gamma_optimal
# gamma_optimal = find_gamma_max(sys_gamma_to_alpha, eq_res= eq_res)
# print("\n Gamma opti : ", gamma_optimal)
    




