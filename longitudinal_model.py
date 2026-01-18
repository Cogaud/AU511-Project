import numpy as np
from math import sin, cos, pi, sqrt, atan, tan
from atm_std import *
from aircraft_data import *
from sisopy31 import *


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

def short_period_linearization(Ar,Br,Cr,Dr):
    # Create the state space system
    Ai = Ar[2:4, 2:4]
    Bi = Br[2:4, 0:1]
    # damp(Ai)  # error function
    Cia = np.matrix ( [ [ 1, 0 ] ] )
    Ciq = np.matrix ( [ [  0, 1 ] ] )
    Di = np.matrix ( [ [ 0 ] ] )
    TaDm_ss= control.ss ( Ai , Bi , Cia , Di )
    print("\n Transfer Function Short Period (alpha/delta_m):")
    TaDm_tf = control.tf ( TaDm_ss )
    print ( TaDm_tf )
    print("\n Static gain of alpha/delta_m:%f" % (control.dcgain(TaDm_tf)))

    TqDm_ss= control.ss ( Ai , Bi , Ciq , Di )
    print("\n Transfer Function Short Period (q/delta_m):")
    TqDm_tf = control.tf ( TqDm_ss )
    print ( TqDm_tf )
    print("\n Static gain of q/delta_m:%f" % (control.dcgain(TqDm_tf)))

    figure(1)
    Ya, Ta = control.matlab.step(TaDm_tf, arange(0, 10, 0.01))
    Yq, Tq = control.matlab.step(TqDm_tf, arange(0, 10, 0.01))
    plot(Ta, Ya, 'b', Tq, Yq, 'r', lw=2)
    plot([0, Ta[-1]], [Ya[-1], Ya[-1]], 'k--', lw=1)
    plot([0, Ta[-1]], [1.05*Ya[-1], 1.05*Ya[-1]], 'k--', lw=1)
    plot([0, Ta[-1]], [0.95*Ya[-1], 0.95*Ya[-1]], 'k--', lw=1)
    plot([0, Tq[-1]], [Yq[-1], Yq[-1]], 'k--', lw=1)
    plot([0, Tq[-1]], [1.05*Yq[-1], 1.05*Yq[-1]], 'k--', lw=1)
    plot([0, Tq[-1]], [0.95*Yq[-1], 0.95*Yq[-1]], 'k--', lw=1)
    minorticks_on()
    # grid(b=True, which='both')
    grid(True)
    title('Step Response Short Period Approximation')
    legend((r'$\alpha/\delta_m$', r'$q/\delta_m$'), loc='best')
    xlabel('Time (s)')
    ylabel(r'$\alpha$ (rad) and $q$ (rad/s)')
    Osa, Tra, Tsa = step_info(Ta, Ya)
    Osq, Trq, Tsq = step_info(Tq, Yq)
    yya = interp1d(Ta, Ya)
    plot(Tsa, yya(Tsa), 'bs')
    text(Tsa, yya(Tsa)-0.2,Tsa)
    yyq = interp1d(Tq, Yq)
    plot(Tsq, yyq(Tsq), 'rs')
    text(Tsq, yyq(Tsq)-0.2,Tsq)
    print('Alpha setting time 5%% = %f s' % Tsa)
    print('q setting time 5%% = %f s' % Tsq) 
    show()

def phugoid_linearization(Ar,Br,Cr,Dr):
    # Create the state space system
    Ap = Ar[0:2, 0:2]
    Bp = Br[0:2, 0:1]
    # damp(Ap)  # error function
    Cpv = np.matrix ( [ [ 1, 0 ] ] )
    Cpg = np.matrix ( [ [  0, 1 ] ] )
    Dp = np.matrix ( [ [ 0 ] ] )
    TvDm_ss= control.ss ( Ap , Bp , Cpv , Dp )
    print("\n Transfer Function Phugoid (V/delta_m):")
    TvDm_tf = control.tf ( TvDm_ss )
    print ( TvDm_tf )
    print("\n Static gain of V/delta_m:%f" % (control.dcgain(TvDm_tf)))

    TgDm_ss= control.ss ( Ap , Bp , Cpg , Dp )
    print("\n Transfer Function Phugoid (gamma/delta_m):")
    TgDm_tf = control.tf ( TgDm_ss )
    print ( TgDm_tf )
    print("\n Static gain of gamma/delta_m:%f" % (control.dcgain(TgDm_tf)))
    figure(1)
    Yv, Tv = control.matlab.step(TvDm_tf, arange(0, 700, 0.01))
    Yg, Tg = control.matlab.step(TgDm_tf, arange(0, 700, 0.01))
    plot(Tv, Yv, 'b', Tg, Yg, 'r', lw=2)
    plot([0, Tv[-1]], [Yv[-1], Yv[-1]], 'k--', lw=1)
    plot([0, Tv[-1]], [1.05*Yv[-1], 1.05*Yv[-1]], 'k--', lw=1)
    plot([0, Tv[-1]], [0.95*Yv[-1], 0.95*Yv[-1]], 'k--', lw=1)
    plot([0, Tg[-1]], [Yg[-1], Yg[-1]], 'k--', lw=1)
    plot([0, Tg[-1]], [1.05*Yg[-1], 1.05*Yg[-1]], 'k--', lw=1)
    plot([0, Tg[-1]], [0.95*Yg[-1], 0.95*Yg[-1]], 'k--', lw=1)
    minorticks_on()
    # grid(b=True, which='both')
    grid(True)
    title('Step Response Phugoid Approximation')
    legend((r'$V/\delta_m$', r'$\gamma/\delta_m$'), loc='best')
    xlabel('Time (s)')
    ylabel(r'$V$ (rad) and $\gamma$ (rad/s)')
    Osv, Trv, Tsv = step_info(Tv, Yv)
    Osg, Trg, Tsg = step_info(Tg, Yg)
    yyv = interp1d(Tv, Yv)
    plot(Tsv, yyv(Tsv), 'bs')
    text(Tsv, yyv(Tsv)-0.2,Tsv)
    yyg = interp1d(Tg, Yg)
    plot(Tsg, yyg(Tsg), 'rs')
    text(Tsg, yyg(Tsg)-0.2,Tsg)
    print('V setting time 5%% = %f s' % Tsv)
    print('gamma setting time 5%% = %f s' % Tsg) 
    show()

def closed_loop_q_qc(Ak, Bk, Ck, Dk):
    figure(4)
    Tqcl_ss= control.ss ( Ak , Bk , Ck , Dk )
    print("\n Closed Loop Transfer Function q/q_c :")
    Tqcl_tf = control.tf ( Tqcl_ss )
    print ( Tqcl_tf )
    
    Yqcl, Tqcl = control.matlab.step(Tqcl_tf, arange(0, 5, 0.01))
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
    Osqcl, Trqcl, Tsqcl = step_info(Tqcl, Yqcl)
    yyqcl = interp1d(Tqcl, Yqcl)
    plot(Tsqcl, yyqcl(Tsqcl), 'rs')
    text(Tsqcl, yyqcl(Tsqcl)-0.02,Tsqcl)
    print(' q Setting time 5%% = %f s' % Tsqcl)
    show()
    return Tqcl_ss, Tqcl_tf

def closed_loop_g_gc_plot(Ag, Bg, Cg, Dg):
    figure(5)
    Tgcl_ss= control.ss ( Ag , Bg , Cg , Dg )
    print("\n Closed Loop Transfer Function gamma/gamma_c :")
    Tgcl_tf = control.tf ( Tgcl_ss )
    print ( Tgcl_tf )

    Ygcl, Tgcl = control.matlab.step(Tgcl_tf, arange(0, 200, 0.01))
    plot(Tgcl, Ygcl, 'b', lw=2)
    plot([0, Tgcl[-1]], [Ygcl[-1], Ygcl[-1]], 'k--', lw=1)
    plot([0, Tgcl[-1]], [1.05*Ygcl[-1], 1.05*Ygcl[-1]], 'k--', lw=1)
    plot([0, Tgcl[-1]], [0.95*Ygcl[-1], 0.95*Ygcl[-1]], 'k--', lw=1)
    minorticks_on()
    # grid(b=True, which='both')
    grid(True)
    title('Step Response $gamma/gamma_c$')
    xlabel('Time (s)')
    ylabel(r'$gamma$ (rad/s)')
    Osgcl, Trgcl, Tsgcl = step_info(Tgcl, Ygcl)
    yyqcl = interp1d(Tgcl, Ygcl)
    plot(Tsgcl, yyqcl(Tsgcl), 'rs')
    text(Tsgcl, yyqcl(Tsgcl)-0.02,Tsgcl)
    print(' gamma Setting time 5%% = %f s' % Tsgcl)
    show()
    return Tgcl_ss, Tgcl_tf

def closed_loop_z_zc_plot(Az, Bz, Cz, Dz):
    figure(6)
    Tzcl_ss= control.ss ( Az , Bz , Cz , Dz )
    print("\n Closed Loop Transfer Function z/z_c :")
    Tzcl_tf = control.tf ( Tzcl_ss )
    print(Tzcl_tf)

    Yzcl, Tzcl = control.matlab.step(Tzcl_tf, arange(0, 20, 0.01))
    plot(Tzcl, Yzcl, 'b', lw=2)
    plot([0, Tzcl[-1]], [Yzcl[-1], Yzcl[-1]], 'k--', lw=1)
    plot([0, Tzcl[-1]], [1.05*Yzcl[-1], 1.05*Yzcl[-1]], 'k--', lw=1)
    plot([0, Tzcl[-1]], [0.95*Yzcl[-1], 0.95*Yzcl[-1]], 'k--', lw=1)
    minorticks_on()
    # grid(b=True, which='both')
    grid(True)
    title('Step Response $z/z_c$')
    xlabel('Time (s)')
    ylabel(r'$z$ (m)')
    Oszcl, Trzcl, Tszcl = step_info(Tzcl, Yzcl)
    yyqcl = interp1d(Tzcl, Yzcl)
    plot(Tszcl, yyqcl(Tszcl), 'rs')
    text(Tszcl, yyqcl(Tszcl)-0.2,Tszcl)
    print(' z Setting time 5%% = %f s' % Tszcl)
    show()
    return Tzcl_ss, Tzcl_tf

# Helpers for Forces
def D_eq(rho, V, S, Cx):
    return 0.5 * rho * V ** 2 * S * Cx


def L_eq(rho, V, S, Cz):
    return 0.5 * rho * V ** 2 * S * Cz


def M_aero(rho, V, S, l_ref, Cm):
    return 0.5 * rho * V ** 2 * S * l_ref * Cm


