import numpy as np
from scipy.interpolate import interp1d

# # deprecated version of data
# def get_aero_coefs(Mach):
#     """
#     Returns aerodynamic coefficients for a given Mach number
#     based on the graphs from the Practical Work PDF.
#     """
#     # Data digitized from PDF graphs (approximate)
#     # Mach breakpoints
#     M_points = np.array([0.0, 0.4, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0])

#     # Cx0 (Drag coeff at null incidence) - Page 9
#     Cx0_data = np.array(
#         [0.015, 0.015, 0.015, 0.017, 0.020, 0.025, 0.030, 0.036, 0.038, 0.035, 0.032, 0.030, 0.029, 0.028])

#     # Cza (Lift gradient) - Page 10
#     Cza_data = np.array([2.7, 2.7, 2.7, 2.75, 2.9, 3.1, 3.3, 3.25, 3.1, 2.8, 2.6, 2.4, 2.2, 2.0])

#     # Czdm (Lift gradient wrt elevator) - Page 11
#     Czdm_data = np.array([1.1, 1.1, 1.1, 1.15, 1.2, 1.25, 1.22, 1.15, 1.0, 0.7, 0.5, 0.42, 0.35, 0.28])

#     # dm0 (Equilibrium fin deflection for null lift) - Page 12
#     # Graph shows 0.022 flat, drops at 0.8 to 0 at 1.2, then negative
#     dm0_data = np.array(
#         [0.022, 0.022, 0.022, 0.022, 0.020, 0.015, 0.010, 0.005, 0.000, 0.000, -0.002, -0.005, -0.012, -0.020])

#     # alpha0 (Incidence for null lift) - Page 13
#     # Graph starts high (0.02 or so), drops sharply at M1.0, then rises
#     alpha0_data = np.array(
#         [0.02, 0.02, 0.02, 0.02, 0.015, 0.005, 0.005, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011])

#     # f (Aerodynamic center position % lt) - Page 14
#     # 0.54 up to 0.85, rises to 0.61 at M1.05
#     f_data = np.array([0.54, 0.54, 0.54, 0.545, 0.56, 0.59, 0.605, 0.608, 0.608, 0.608, 0.608, 0.608, 0.608, 0.608])

#     # f_delta (Fins aerodynamic center) - Page 15
#     # 0.78 up to 0.85, rises to 0.9 at M1.2
#     f_delta_data = np.array([0.78, 0.78, 0.78, 0.79, 0.81, 0.86, 0.88, 0.89, 0.895, 0.9, 0.9, 0.9, 0.9, 0.9])

#     # k (Polar coefficient) - Page 16
#     # 0.22 up to 0.8, linear increase after
#     k_data = np.array([0.22, 0.22, 0.22, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.40, 0.45, 0.50, 0.55])

#     # Cmq (Damping coefficient) - Page 17
#     # -0.68 flat, spikes up at M0.95, then settles
#     Cmq_data = np.array([-0.68, -0.68, -0.68, -0.68, -0.7, -0.6, -0.25, -0.3, -0.4, -0.42, -0.38, -0.32, -0.28, -0.26])

#     # Interpolation
#     kind = 'linear'

#     # Helper for safe interpolation
#     def get_val(x, y, val):
#         if val <= x[0]: return y[0]
#         if val >= x[-1]: return y[-1]
#         return interp1d(x, y, kind=kind)(val)

#     coefs = {}
#     coefs['Cx0'] = float(get_val(M_points, Cx0_data, Mach))
#     coefs['Cza'] = float(get_val(M_points, Cza_data, Mach))
#     coefs['Czdm'] = float(get_val(M_points, Czdm_data, Mach))
#     coefs['dm0'] = float(get_val(M_points, dm0_data, Mach))
#     coefs['alpha0'] = float(get_val(M_points, alpha0_data, Mach))
#     coefs['f'] = float(get_val(M_points, f_data, Mach))
#     coefs['f_delta'] = float(get_val(M_points, f_delta_data, Mach))
#     coefs['k'] = float(get_val(M_points, k_data, Mach))
#     coefs['Cmq'] = float(get_val(M_points, Cmq_data, Mach))

#     return coefs


"""
Those value are the one from graphics at an altitude of 11812 ft and Mach 0.95
"""

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
m = 8400.0                      # Mass (kg)
S = 34.0                        # Reference Surface (m^2)
l_ref = 5.24                    # Mean aerodynamic chord (m)
l_t = (3/2) * l_ref             # Tail arm (m)
c = 0.52                        # (52 %)
r_g = 2.65                      # Radius of gyration (m)
Iy = m * r_g ** 2               # Inertia Moment y-axis (kg.m^2)
G = -c*l_t                       #
F = -coefs['f'] * l_t            #
F_delta = -coefs['f_d'] * l_t
X = F - G 
Y = F_delta - G