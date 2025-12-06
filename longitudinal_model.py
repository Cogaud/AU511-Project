import numpy as np
from math import sin, cos, pi, sqrt, atan, tan
from atm_std import *
from aircraft_data import *


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
    print(f"  Cma:     {Cma:.4f}")
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
    mq = (Q * S * l_ref**2 * Cmq)/(V_eq * Iy)
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


