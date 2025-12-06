# --- Main Execution ---
import numpy as np
from equilibrum import find_equilibrium_point52
from longitudinal_model import get_state_space
from analize import analyze_modes,calculate_short_period_approximation, calculate_phugoid_approximation, transfert_function, ploting_step_response, ploting_bode

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

    # 5. Phugoid analysis
    calculate_phugoid_approximation(A)

    # 6. Transfer function
    tf = transfert_function(A, B, C, D)
    print("\nTransfer Function:")
    print(tf)

    # 7. Step response
    ploting_step_response(A, B, C, D)

    # 8. Bode plot
    ploting_bode(A, B, C, D)