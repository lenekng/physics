"""Calculation of eigenvalues and eigenfunctions of the 1D Schrödinger equation.
"""

import numpy as np
from scipy.linalg import eigh


def discretization(xmin, xmax, N, retstep=False):
    """Compute the quantum-mechanically correct spatial discretization.

    Parameters:
        xmin: lower bound of the interval
        xmax: upper bound of the interval
        N: number of discretization points
        retstep: decides whether the step size is returned
    Returns:
        x: array of discretized position points
        delta_x (only if `retstep` is True): spatial grid spacing
    """
    delta_x = (xmax - xmin) / (N + 1)                      # spatial grid spacing
    x = np.linspace(xmin + delta_x, xmax - delta_x, N)    # spatial grid points

    if retstep:
        return x, delta_x
    else:
        return x


def diagonalization(hbar_eff, x, V):
    """Compute sorted eigenvalues and corresponding eigenfunctions.

    Parameters:
        hbar_eff: effective reduced Planck constant
        x: position points
        V: potential as a function of one variable
    Returns:
        ew: sorted eigenvalues (array of length N)
        ef: corresponding eigenvectors, ef[:, i] (size N*N)
    """
    delta_x = x[1] - x[0]
    v_values = V(x)                                       # potential values

    N = len(x)
    z = hbar_eff**2 / (2.0 * delta_x**2)                  # off-diagonal element
    h = (np.diag(v_values + 2.0 * z) +
         np.diag(-z * np.ones(N - 1), k=-1) +             # matrix representation
         np.diag(-z * np.ones(N - 1), k=1))               # Hamiltonian operator

    ew, ef = eigh(h)                                      # diagonalization
    ef = ef / np.sqrt(delta_x)                            # wavefunction normalization
    return ew, ef


def plot_eigenfunctions(ax, ew, ef, x, V, width=1, Emax=0.15, scale=0.01,
                        magnitude_squared=False, baseline=True, alpha=1.0,
                        title=None):
    """Plot the eigenfunctions.

    The lowest eigenfunctions 'ef' in the potential 'V'(x) are plotted
    at the level of the eigenvalues 'ew' in the plotting area 'ax'
    (provided in the calling program, for example by
    ``ax = fig.add_subplot(111)``).
    The eigenvalues are assumed to be sorted.

    Optional parameters:
        width: (default value 1) specifies the line width for plotting the
            eigenfunctions. width can also be an array of line widths
            with a specific value for each eigenfunction.
        Emax: (default value 0.15) sets the upper energy limit
            for the plot.
        scale: scaling factor for the graphical representation
            of the eigenfunctions.
        magnitude_squared: specifies whether the magnitude squared of the
            eigenfunction or the (real!) eigenfunction itself is plotted.
        baseline: specifies whether a dashed gray line is drawn
            at the level of each eigenenergy.
        alpha: specifies the transparency when plotting the eigenfunctions
            (see also the Matplotlib documentation for plot()). alpha can
            also be an array of transparency values with a specific value
            for each eigenfunction.
        title: title of the plot.
    """
    if title is None:
        title = "Asymmetric double-well potential"

    v_values = V(x)                                  # potential values

    # configure position-space plot window
    ax.autoscale(False)
    ax.axis([np.min(x), np.max(x), np.min(v_values), Emax])
    ax.set_xlabel(r'$x$')
    ax.set_title(title)

    ax.plot(x, v_values, linewidth=2, color='0.7')  # plot potential
    n_states = np.sum(ew <= Emax)                   # number of eigenfunctions to plot

    if baseline:                                    # plot baseline at eigenvalues
        for i in range(n_states):
            ax.plot(x, ew[i] + np.zeros(len(x)), ls='--', color='0.7')

    try:                                            # does width behave
        iter(width)                                 # like an array?
    except TypeError:                               # if `width` is scalar:
        width = width * np.ones(n_states)           # constant line width

    try:                                            # similarly for
        iter(alpha)                                 # transparency alpha
    except TypeError:
        alpha = alpha * np.ones(n_states)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']         # fixed color order
    if magnitude_squared:                           # plot magnitude squared of eigenfunction
        ax.set_ylabel(r'$V(x)\ \rm{,\ \|Efkt.\|^{2}\ at\ eigenvalue}$')
        for i in range(n_states):
            ax.plot(x, ew[i] + scale * np.abs(ef[:, i])**2, linewidth=width[i],
                    color=colors[i % len(colors)], alpha=alpha[i])
    else:                                           # plot eigenfunction
        ax.set_ylabel(r'$V(x)\ \rm{,\ eigenfunction\ at\ eigenvalue}$')
        for i in range(n_states):
            ax.plot(x, ew[i] + scale * ef[:, i], linewidth=width[i],
                    color=colors[i % len(colors)], alpha=alpha[i])