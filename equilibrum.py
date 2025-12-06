import numpy as np
from math import sin, cos, pi, sqrt, atan, tan
from atm_std import *
from aircraft_data import *

# --- 1. Equilibrium Finder ---

def find_equilibrium_point52(Mach, Altitude_ft):
    """
    Solves for alpha_eq, delta_m_eq, and Thrust_eq
    based on the algorithm in Slide 21.
    """
    Altitude_m = Altitude_ft * 0.3048
    _, rho, a = get_cte_atm(Altitude_m)
    V_eq = Mach * a
    Q = 0.5 * rho * V_eq ** 2

    # Unpack coefficients
    Cx0 = coefs['C_x_0']
    k = coefs['k']
    Cza = coefs['C_z_a']
    Czdm = coefs['C_z_d_m']
    dm0 = coefs['d_m0']
    alpha0 = coefs['alpha_0']
    f = coefs['f']
    fd = coefs['f_d']

    # Iteration initialization
    alpha_eq = 0.0  # rad
    gamma_eq = 0.0  # rad (Level flight assumption for operating point)
    Thrust_eq = 0.0  # N

    # Loop parameters
    tolerance = 1e-7
    max_iter = 100
    diff = 1.0
    iter_count = 0

    print(f"\n--- Finding Equilibrium for M={Mach}, Z={Altitude_ft}ft ---")

    while diff > tolerance and iter_count < max_iter:
        # 1. Calculate required Lift Coefficient (Cz_eq)
        # Assuming level flight (gamma=0), Lift + Thrust*sin(alpha) = Weight
        # Algorithm slide 21: Cz_eq = (1/QS) * (m*g0 - Fpx * sin(alpha))
        # Approximation for first step: Fpx (Thrust) is unknown, assume small alpha contribution
        # For better convergence, we solve the coupled system:
        # Lift = Q * S * Cz
        # Drag = Q * S * Cx
        # Thrust * cos(alpha) = Drag
        # Lift + Thrust * sin(alpha) = m * g0

        # Step 1: Current Lift estimate 
        Cz_eq = (m * g0 - Thrust_eq * sin(alpha_eq)) / (Q * S)

        # 2. Right coefficients
        Cx_eq = Cx0 + k * Cz_eq ** 2
        Cxdm = 2 * k * Cz_eq * Czdm

        # 4. Moment Equilibrium to find delta_m
        numerator = Cx_eq * sin(alpha_eq) + Cz_eq * cos(alpha_eq)
        denominator = Cxdm * sin(alpha_eq) + Czdm * cos(alpha_eq)
        
        # delta_m_eq = dm0 - (numerator / denominator) * X / (Y - X)
        delta_m_eq = dm0 - (numerator / denominator) * X / (Y - X)

        # 5. Update alpha using Lift relation
        alpha_next = alpha0 + (Cz_eq - Czdm * delta_m_eq) / Cza
        Thrust_next = (Q * S * Cx_eq) / cos(alpha_eq)

        diff = abs(alpha_next - alpha_eq)
        alpha_eq = alpha_next
        Thrust_eq = Thrust_next
        iter_count += 1

    results = {
        "Mach": Mach,
        "V_eq": V_eq,
        "rho": rho,
        "alpha_eq_deg": np.degrees(alpha_eq),
        "alpha_eq_rad": alpha_eq,
        "delta_m_eq_deg": np.degrees(delta_m_eq),
        "delta_m_eq_rad": delta_m_eq,
        "gamma_eq": gamma_eq,
        "theta_eq": alpha_eq + gamma_eq,
        "Thrust": Thrust_eq,
        "Cz_eq": Cz_eq,
        "Cx_eq": Cx_eq,
        "coefs": coefs
    }
    return results