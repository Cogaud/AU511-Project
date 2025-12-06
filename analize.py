import numpy as np
from math import sin, cos, pi, sqrt, atan, tan

from __future__ import unicode_literals
from matplotlib.pyplot import *
import control
from control.matlab import *
from math import *
from scipy.interpolate import interp1d
from pylab import *
from matplotlib . widgets import Slider
import numpy as np
from atm_std import *
import scipy.interpolate
from sisopy31 import *
from matplotlib . pylab import *
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

def eig_analysis(A):
    eigenvalues, eigenvectors  = np.linalg.eig(A)
    return eigenvalues, eigenvectors


# --- Short Period Approximation ---
