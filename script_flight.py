import numpy as np
import matplotlib.pyplot as plt
from main import eq_res, sys_g_dm, sys_z_dm
from scipy.integrate import odeint

# =============================
# Multi-Phase Flight Simulation
# =============================
# Using:
# - sys_g_dm: gamma hold control loop (input: gamma_desired, output: [gamma, alpha, q, theta, z])
# - sys_z_dm: altitude hold control loop (input: z_desired, output: [gamma, alpha, q, theta, z])

def simulate_phase(sys, setpoint, duration, dt, initial_state, phase_name):
    """
    Simulates one flight phase
    
    Parameters:
    -----------
    sys : control.StateSpace
        System to simulate (sys_g_dm or sys_z_dm)
    setpoint : float
        Desired value (gamma in rad or z in m)
    duration : float
        Phase duration in seconds
    dt : float
        Time step in seconds
    initial_state : ndarray
        Initial state vector [gamma, alpha, q, theta, z]
    phase_name : str
        Name of the phase for display
    
    Returns:
    --------
    time : ndarray
        Time vector
    states : ndarray
        State history (n_steps x 5)
    altitude : ndarray
        Altitude history
    """
    
    n_steps = int(duration / dt)
    time = np.arange(0, duration, dt)
    
    # State vector: [gamma, alpha, q, theta, z]
    states = np.zeros((n_steps, 5))
    states[0] = initial_state
    
    # System matrices
    A = sys.A
    B = sys.B
    
    print(f"\n{phase_name}:")
    print(f"  Duration: {duration}s, Setpoint: {setpoint:.4f}")
    
    # Simulate using Euler method
    for i in range(1, n_steps):
        # Control input is the setpoint
        u = np.array([[setpoint]])
        
        # State derivative: dx/dt = Ax + Bu
        dx = A @ states[i-1].reshape(5, 1) + B @ u
        
        # Update state
        states[i] = states[i-1] + (dx.flatten() * dt)
    
    altitude = states[:, 4]  # z is the 5th state (index 4)
    
    print(f"  Initial altitude: {altitude[0]:.2f} m")
    print(f"  Final altitude: {altitude[-1]:.2f} m")
    print(f"  Final gamma: {np.degrees(states[-1, 0]):.4f}°")
    
    return time, states, altitude


def multi_phase_flight_simulation():
    """
    Complete flight simulation with all phases as per requirements
    """
    
    # Get systems and equilibrium data from main.py
    # from main import sys_g_dm, sys_z_dm, eq_res

    # Initial conditions at sea level
    gamma_0 = 0.0          # rad (level flight)
    alpha_0 = eq_res['alpha_eq_rad']
    q_0 = 0.0              # rad/s
    theta_0 = eq_res['theta_eq']
    z_0 = 0.0              # m (sea level)
    
    initial_state = np.array([gamma_0, alpha_0, q_0, theta_0, z_0])
    
    dt = 0.1  # Time step in seconds
    all_times = []
    all_altitudes = []
    all_gammas = []
    all_alphas = []
    all_qs = []
    all_thetas = []
    current_time = 0
    
    # ============================================
    # PHASE 1: ASCENT - Constant Flight Path Angle
    # ============================================
    gamma_ascent = np.radians(5.0)  # 5 degrees climb
    duration_ascent = 200  # seconds
    
    t1, states1, alt1 = simulate_phase(sys_g_dm, gamma_ascent, duration_ascent, dt, 
                                       initial_state, "PHASE 1: ASCENT (γ hold)")
    
    all_times.extend(t1 + current_time)
    all_altitudes.extend(alt1)
    all_gammas.extend(states1[:, 0])
    all_alphas.extend(states1[:, 1])
    all_qs.extend(states1[:, 2])
    all_thetas.extend(states1[:, 3])
    
    current_time = all_times[-1]
    final_state_phase1 = states1[-1]
    cruise_altitude = alt1[-1]
    
    # ============================================
    # PHASE 2: CRUISE - Constant Altitude
    # ============================================
    duration_cruise = 100  # seconds (~100s as per requirements)
    
    t2, states2, alt2 = simulate_phase(sys_z_dm, cruise_altitude, duration_cruise, dt,
                                       final_state_phase1, "PHASE 2: CRUISE (z hold)")
    
    all_times.extend(t2 + current_time)
    all_altitudes.extend(alt2)
    all_gammas.extend(states2[:, 0])
    all_alphas.extend(states2[:, 1])
    all_qs.extend(states2[:, 2])
    all_thetas.extend(states2[:, 3])
    
    current_time = all_times[-1]
    final_state_phase2 = states2[-1]
    
    # ============================================
    # PHASE 3: DESCENT - Constant Flight Path Angle
    # ============================================
    gamma_descent = np.radians(-3.0)  # -3 degrees descent
    duration_descent = 250  # seconds
    
    t3, states3, alt3 = simulate_phase(sys_g_dm, gamma_descent, duration_descent, dt,
                                       final_state_phase2, "PHASE 3: DESCENT (γ hold)")
    
    all_times.extend(t3 + current_time)
    all_altitudes.extend(alt3)
    all_gammas.extend(states3[:, 0])
    all_alphas.extend(states3[:, 1])
    all_qs.extend(states3[:, 2])
    all_thetas.extend(states3[:, 3])
    
    current_time = all_times[-1]
    final_state_phase3 = states3[-1]
    final_altitude_phase3 = alt3[-1]
    
    # ============================================
    # PHASE 4: FLARE & LEVEL FLIGHT
    # ============================================
    # Short flare phase then level flight at low altitude
    duration_flare = 60  # seconds
    
    t4, states4, alt4 = simulate_phase(sys_z_dm, final_altitude_phase3, duration_flare, dt,
                                       final_state_phase3, "PHASE 4: FLARE & LEVEL FLIGHT (z hold)")
    
    all_times.extend(t4 + current_time)
    all_altitudes.extend(alt4)
    all_gammas.extend(states4[:, 0])
    all_alphas.extend(states4[:, 1])
    all_qs.extend(states4[:, 2])
    all_thetas.extend(states4[:, 3])
    
    # Convert to arrays
    all_times = np.array(all_times)
    all_altitudes = np.array(all_altitudes)
    all_gammas = np.array(all_gammas)
    all_alphas = np.array(all_alphas)
    all_qs = np.array(all_qs)
    all_thetas = np.array(all_thetas)
    
    return all_times, all_altitudes, all_gammas, all_alphas, all_qs, all_thetas


def plot_flight_path(times, altitudes, gammas, alphas, qs, thetas):
    """
    Plots the complete flight profile
    Returns two separate figures: one for altitude path, one for other parameters
    """
    
    # FIGURE 1: Flight Path (Altitude vs Time) - MAIN WINDOW
    fig1 = plt.figure(figsize=(14, 8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(times, altitudes, 'b-', linewidth=3)
    ax1.fill_between(times, 0, altitudes, alpha=0.2, color='blue')
    ax1.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Altitude (m)', fontsize=13, fontweight='bold')
    ax1.set_title('Flight Path - Altitude vs Time', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark phases
    ax1.axvspan(0, 200, alpha=0.1, color='green', label='Ascent (γ hold)')
    ax1.axvspan(200, 300, alpha=0.1, color='blue', label='Cruise (z hold)')
    ax1.axvspan(300, 550, alpha=0.1, color='red', label='Descent (γ hold)')
    ax1.axvspan(550, 610, alpha=0.1, color='orange', label='Flare/Level (z hold)')
    ax1.legend(loc='best', fontsize=11)
    
    fig1.tight_layout()
    
    # FIGURE 2: Other Parameters (2x3 grid)
    fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle('Multi-Phase Flight Management - Complete Profile', fontsize=16, fontweight='bold')
    
    # Plot 1: Flight Path Angle vs Time
    ax2 = axes[0, 0]
    ax2.plot(times, np.degrees(gammas), 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Flight Path Angle (deg)', fontsize=11)
    ax2.set_title('Flight Path Angle (γ) vs Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot 2: Angle of Attack vs Time
    ax3 = axes[0, 1]
    ax3.plot(times, np.degrees(alphas), 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Angle of Attack (deg)', fontsize=11)
    ax3.set_title('Angle of Attack (α) vs Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 3: Angle of Attack vs Flight Path Angle
    ax4 = axes[0, 2]
    ax4.plot(np.degrees(gammas), np.degrees(alphas), 'purple', linewidth=2)
    ax4.set_xlabel('Flight Path Angle (deg)', fontsize=11)
    ax4.set_ylabel('Angle of Attack (deg)', fontsize=11)
    ax4.set_title('α vs γ (Phase Plane)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 4: Pitch Rate vs Time
    ax5 = axes[1, 0]
    ax5.plot(times, np.degrees(qs), 'purple', linewidth=2)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Pitch Rate (deg/s)', fontsize=11)
    ax5.set_title('Pitch Rate (q) vs Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot 5: Pitch Attitude vs Time
    ax6 = axes[1, 1]
    ax6.plot(times, np.degrees(thetas), 'brown', linewidth=2)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Pitch Attitude (deg)', fontsize=11)
    ax6.set_title('Pitch Attitude (θ) vs Time', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 6: Vertical Speed (Rate of Climb)
    ax7 = axes[1, 2]
    vertical_speed = np.gradient(altitudes, times)
    ax7.plot(times, vertical_speed, 'darkblue', linewidth=2)
    ax7.set_xlabel('Time (s)', fontsize=11)
    ax7.set_ylabel('Vertical Speed (m/s)', fontsize=11)
    ax7.set_title('Vertical Speed (dz/dt) vs Time', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    fig2.tight_layout()
    return fig1, fig2


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("MULTI-PHASE FLIGHT SIMULATION")
    print("="*60)
    
    # Run simulation
    times, altitudes, gammas, alphas, qs, thetas = multi_phase_flight_simulation()
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total simulation time: {times[-1]:.1f} seconds")
    print(f"Maximum altitude reached: {np.max(altitudes):.2f} m")
    print(f"Final altitude: {altitudes[-1]:.2f} m")
    print(f"Total altitude change: {altitudes[-1] - altitudes[0]:.2f} m")
    
    # Plot results (two separate windows)
    fig1, fig2 = plot_flight_path(times, altitudes, gammas, alphas, qs, thetas)
    
    # Save figures
    plt.figure(fig1.number)
    plt.savefig('flight_path_altitude.png', dpi=150, bbox_inches='tight')
    print("\nFigure 1 saved as 'flight_path_altitude.png'")
    
    plt.figure(fig2.number)
    plt.savefig('flight_management_profile.png', dpi=150, bbox_inches='tight')
    print("Figure 2 saved as 'flight_management_profile.png'")
    
    plt.show()
