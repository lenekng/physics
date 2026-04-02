"""
Determination of the eigenvalues and eigenfunctions of a particle in an
asymmetric double-well potential V(x) = x^4 - x^2 - A*x.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def potential_harm(x):
    """Potential of the harmonic oscillator. x is the position.
    """
    return 1 / 2 * x**2

def double_well(x, A):
    """
    Double-well potential. x is the position and A is a parameter.
    """
    return x**4 - x**2 - A * x

def ew_ev(potential, h_eff, x_min, A, N):
    """Determination of the eigenvalues and eigenfunctions for the potential
    potential. h_eff and A are dimensionless parameters. N is the size of the
    matrix, x_min is the starting point of the interval. A symmetric interval
    around 0 is considered here. Therefore, x_max follows from the choice of
    x_min.
    """
    x_max = -x_min
    dx = (x_max - x_min)/(N + 1)        # interval width
    print('Interval width = {:.2f}'.format(dx))

    x_i = x_min + np.arange(N) * dx     # position
    z = np.ones(N) * h_eff**2 / (2 * dx**2)

    # Off-diagonal terms, otherwise the line becomes too long
    off1 = np.diag(-z[:-1], k=-1)
    off2 = np.diag(-z[:-1], k=1)
    # Construction of the matrix
    matrix = off1 + off2 + np.diag(potential(x_i, A) + 2*z)

    # Calculation of eigenvalues and eigenvectors
    ew, ev = eigh(matrix)

    # Approximation that the potential is infinite at x_max and x_min
    ev[-1,:] = 0
    ev[0,:] = 0
    return ew, ev/np.sqrt(dx), x_i

def visualization(potential, ax1, ew, ev, x_i, E_max, scale, colors):
    """Visualization of the eigenfunctions and eigenenergies in the potential
    potential. ax1 is the plot window, ew the eigenvalues, ev the
    eigenenergies, x_i the discrete position points, and colors a list.
    """
    i = 0

    # Plot of the eigenfunctions and eigenenergies for E_n < 0.1
    while ew[i] < E_max:
        ax1.plot(x_i, scale * ev.T[i] + ew[i], c=colors[i],
                 label='Eigenfunction {} at eigenenergy {:.3f}'
                 .format(i, ew[i]))
        ax1.axhline(y=ew[i], c=colors[i], ls='dashdot', alpha=0.7)
        i = i + 1

    ax1.plot(x_i, potential, label='Potential')   # plot potential
    ax1.set_xlabel("r")
    ax1.set_ylabel("E")
    ax1.set_title("Eigenfunctions and eigenenergies")
    ax1.legend()

def main():
    """Main program."""
    print(__doc__)

    # Definition of the parameters
    A = 0.06
    h_eff = 0.07
    N = 250                  # size of the matrix
    x_min = -1.5             # interval boundary
    V = double_well          # potential
    scale = 0.01             # scaling factor
    E_max = 0.1              # plot eigenfunctions up to this energy

    # Calculation of the eigenvalues and eigenfunctions
    ew, ev, x_i = ew_ev(V, h_eff, x_min, A, N)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(-0.3, E_max)

    # List for color coding
    colors = ['g', 'orange', 'r', 'c', 'deeppink', 'm', 'y', 'maroon']

    visualization(V(x_i, A), ax1, ew, ev, x_i, E_max, scale, colors)

    # Output of initial values
    print('A = {}, h_eff = {}, N = {}, x_min = {}, x_max = {}, scale = {}'
          .format(A, h_eff, N, x_min, -x_min, scale))

    plt.show()
    
if __name__ == "__main__":
    main()


