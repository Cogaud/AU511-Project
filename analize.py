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



Ar=np . matrix ([
    [0.0146 ,0.0362 ,0.0011 ,0],
    [0.0716 ,0 ,0.7884 ,0],
    [0.0716 ,0 ,0.7884 ,1.0000],
    [0 ,0 ,13.2258 ,0.7808]
])

damp( Ar )

Br=np . matrix ([ [0 ,0.1798 , 0.1798 , 13.7335]]) . T
eigenValues , eigenVectors=np.linalg.eig ( Ar )
print(" Eigenvalues of Ar ")
print ( eigenValues )
print(" Eigenvectors of Ar ")
print ( eigenVectors )

############################# Short period mode
def short_period_mode(Ar, Br):
    Ai=Ar [ 2 : 4 , 2 : 4 ]
    Bi=Br [ 2 : 4 , 0 : 1 ]
    damp( Ai )
    Cia=np . matrix ( [ [ 1 , 0 ] ] )
    Ciq=np . matrix ( [ [ 0 , 1 ] ] )
    Di=np . matrix ( [ [ 0 ] ] )
    TaDmss= control.ss ( Ai , Bi , Cia , Di )
    print ( " Transfer function alpha / delta m = " )
    TaDmtf= control.tf (TaDmss )
    print ( TaDmtf )
    print ( " Static gain of alpha / delta m =%f "%(control.dcgain(TaDmtf)))
    TqDmss= control.ss ( Ai , Bi , Ciq , Di )
    print ( " Transfer function q / delta m =" )
    TqDmtf= control.ss2tf (TqDmss )
    print ( TqDmtf )
    print ( " Static gain of q / del ta m =%f "%(dcgain(TqDmtf)))
    figure ( 1 )
    Ya , Ta= control.matlab.step ( TaDmtf , arange (0 ,10 ,0.01) )
    Yq , Tq= control.matlab.step ( TqDmtf , arange (0 ,10 ,0.01) )
    plot(Ta ,Ya , 'b' ,Tq ,Yq ,  'r' , lw=2)
    plot([ 0 , Ta [ 1 ] ] , [Ya[1] ,Ya[ 1] ] , 'k--' , lw=1)
    plot([ 0 , Ta [ 1 ] ] , [ 1.05 *Ya[1] ,1.05*Ya[1] ] , 'k--' , lw=1)
    plot([ 0 , Ta [ 1 ] ] , [ 0.95 *Ya[1] ,0.95*Ya[1] ] , 'k--' , lw=1)
    plot([ 0 , Ta [ 1 ] ] , [Yq[1] ,Yq[1] ] , 'k--' , lw=1)
    plot([ 0 , Ta [ 1 ] ] , [ 1.05 *Yq[1] ,1.05*Yq[1] ] , 'k--' , lw=1)
    plot([ 0 , Ta [ 1 ] ] , [ 0.95 *Yq[1] ,0.95*Yq[1] ] , 'k--' , lw=1)
    minorticks_on( )
    grid( b=True , which= 'both' )
    # grid ( True )
    title( r' Step response $\alpha/\delta_m$ et $q/\ delta_m$ ' )
    legend(' alpha/delta_m ' , ' q / delta_m ' )
    xlabel( ' Time ( s ) ' )
    ylabel( 'alpha ( rad ) & q ( rad / s ) ' )
    Osa , Tra , Tsa= stepinfo (Ta ,Ya)
    Osq , Trq , Tsq= stepinfo (Tq ,Yq)
    yya=interp1d (Ta ,Ya)
    plot ( Tsa , yya ( Tsa ) ,  'bs'  )
    text ( Tsa , yya ( Tsa )-0.2,Tsa )
    yyq=interp1d (Tq ,Yq)
    plot ( Tsq , yyq( Tsq ) ,  'rs ' )
    text ( Tsq , yyq( Tsq )-0.2 , Tsq )
    print ( ' Alpha Settling time 5%% = %f s '%Tsa )
    print ( ' q Settling time 5%% = %f s '%Tsq )