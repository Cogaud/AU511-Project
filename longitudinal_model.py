import numpy as np
from math import sin, cos, pi, sqrt, atan, tan
from atm_std import *
#import aircraft_data

coefs = {}
coefs['C_x_0'] = 0.024
coefs['C_z_a'] = 3.3
coefs['C_z_d_m'] = 1.25
coefs['d_m0'] = 0.015
coefs['alpha_0'] = 0.01 #rad
coefs['f'] = 0.58
coefs['f_d'] = 0.84
coefs['k'] = 0.27
coefs['C_m_q'] = -0.25

# --- Aircraft Characteristics (Page 5) ---
m = 8400.0  # Mass (kg)
S = 34.0  # Reference Surface (m^2)
l_ref = 5.24  # Mean aerodynamic chord (m)
l_t = (3/2) * l_ref  # Tail arm (m)
c = 0.52 # (52 %)
r_g = 2.65  # Radius of gyration (m)
Iy = m * r_g ** 2  # Inertia Moment y-axis (kg.m^2)
# xcg_percent = 0.52  # Center of gravity position (52%)
# xcg = xcg_percent  # Used in equations relative to length units of f and f_delta
G = c*l_t
F = coefs['f'] * l_t
F_delta = coefs['f_d'] * l_t
X = F - G 
Y = F_delta - G


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


# --- 2. State Space Generation ---

def get_state_space(eq_data):
    """
    Constructs the Linearized State Space Matrices A, B, C, D
    State Vector X = [V, gamma, alpha, q, theta, z]
    Control Vector U = [delta_m]
    """

    # Unpack equilibrium data
    V_eq = eq_data['V_eq']
    rho = eq_data['rho']
    alpha_eq = eq_data['alpha_eq_rad']
    gamma_eq = eq_data['gamma_eq']
    theta_eq = eq_data['theta_eq']
    F_eq = eq_data['Thrust']
    Cx_eq = eq_data['Cx_eq']
    Cz_eq = eq_data['Cz_eq']
    dm_eq = eq_data['delta_m_eq_rad']

    coefs = eq_data['coefs']
    Cx0 = coefs['C_x_0']
    k = coefs['k']
    Cza = coefs['C_z_a']
    Czdm = coefs['C_z_d_m']
    Cmq = coefs['C_m_q']
    f = coefs['f']
    fd = coefs['f_d']

    Cxa = 2 * k * Cz_eq * Cza
    Cma = (X/l_ref) * (Cxa * sin(alpha_eq) + Cza * cos(alpha_eq))
    Cxdm = 2 * k * Cz_eq * Czdm
    Cm_dm = (Y/l_ref) * (Cxdm * sin(alpha_eq) + Czdm * cos(alpha_eq))
    # Dynamic Pressure
    Q = 0.5 * rho * V_eq ** 2

    # --- Simplified longitudinal model ---
    # Composant X
    Xv = (2 * Q * S * Cx_eq) / (m * V_eq)
    Xa = (F_eq/(m * V_eq)) * sin(alpha_eq) + (Q * S * Cxa)/(m * V_eq)
    Xg = (g0 * cos(gamma_eq))/V_eq
    
    # Composant Z
    Zv = (2 * g0) / (V_eq)
    Za = (F_eq/(m * V_eq)) * cos(alpha_eq) + (Q * S * Cza)/(m * V_eq)
    Zdm = (Q * S * Czdm)/(m * V_eq)

    # Composant m
    mv = 0
    ma = (Q * S * l_ref * Cma)/Iy
    mq = (Q * S * l_ref**2 * Cmq)/Iy
    mdm = (Q * S * l_ref * Cm_dm)/Iy

    # --- Equations of Motion Linearization ---
    # x1 = V
    # x2 = gamma
    # x3 = alpha
    # x4 = q
    # x5 = theta
    # x6 = z

    # Construct Matrix A (6x6)
    A = np.array([
        [-Xv, -Xg, -Xa, 000, 000, 000],
        [ Zv, 000,  Za, 000, 000, 000],
        [-Zv, 000, -Za,   1, 000, 000],
        [000, 000,  ma,  mq, 000, 000],
        [000, 000, 000,   1, 000, 000],
        [000, V_eq, 000, 000, 000, 000]
    ])

    # Construct Matrix B (6x1)
    B = np.array([
        [0],
        [Zdm],
        [-Zdm],
        [mdm],
        [0],
        [0]
    ])
    # Construct Matrix C (Output Matrix) - Usually Identity for full state feedback analysis
    # Or specific outputs. Let's output all states.
    C = np.eye(6)

    # Construct Matrix D
    D = np.zeros((6, 1))

    return A, B, C, D


# Helpers for Forces
def D_eq(rho, V, S, Cx):
    return 0.5 * rho * V ** 2 * S * Cx


def L_eq(rho, V, S, Cz):
    return 0.5 * rho * V ** 2 * S * Cz


def M_aero(rho, V, S, l_ref, Cm):
    return 0.5 * rho * V ** 2 * S * l_ref * Cm


# --- 3. Mode Analysis ---

def analyze_modes(A):
    eigenvalues = np.linalg.eigvals(A)
    print("\n--- Open Loop Modes Analysis ---")

    # Filter for complex pairs (Oscillatory modes)
    # Usually 2 pairs: Short Period (fast, well damped) and Phugoid (slow, poorly damped)
    # And real poles (subsidence or spiral, though usually longitudinal is 2 pairs)

    # Sort by frequency (imaginary part)

    print(f"{'Mode':<15} | {'Eigenvalue':<25} | {'Freq (rad/s)':<12} | {'Damping':<10}")
    print("-" * 70)

    processed = [False] * len(eigenvalues)

    for i in range(len(eigenvalues)):
        if processed[i]: continue

        eig = eigenvalues[i]
        if abs(eig.imag) > 1e-6:
            # Complex pair
            omega_n = sqrt(eig.real ** 2 + eig.imag ** 2)
            damping = -eig.real / omega_n
            print(f"{'Oscillatory':<15} | {eig:.4f}        | {omega_n:.4f}       | {damping:.4f}")

            # Mark conjugate as processed
            for j in range(i + 1, len(eigenvalues)):
                if abs(eigenvalues[j].imag + eig.imag) < 1e-5 and abs(eigenvalues[j].real - eig.real) < 1e-5:
                    processed[j] = True
        else:
            # Real pole
            print(f"{'Real':<15} | {eig.real:.4f} + 0j           | {'-':<12} | {'-':<10}")

    return eigenvalues


# --- Main Execution ---

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