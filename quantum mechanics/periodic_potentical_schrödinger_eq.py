"""
Calculation of eigenenergies and eigenfunctions of a particle in a
periodic potential: V(x) = A * cos(2 * pi * x).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import functools

def periodic_potential(A, x):
    """ Periodic potential """
    return A * np.cos(2 * np.pi * x)

def discretization(xmin, xmax, N, retstep=False):
    """Compute the quantum-mechanically correct spatial discretization.

    Parameters:
        xmin: lower bound of the domain
        xmax: upper bound of the domain
        N: number of discretization points
        retstep: decides whether the step size is returned

    Returns:
        x: array of discretized spatial points
        delta_x (only if `retstep` is True): grid spacing
    """
    delta_x = (xmax - xmin) / (N)
    x = np.linspace(xmin, xmax - delta_x, N)

    if retstep:
        return x, delta_x
    else:
        return x

def diagonalization(hbar_eff, x, V, k):
    """Compute sorted eigenvalues and corresponding eigenfunctions.

    Parameters:
        hbar_eff: effective hbar
        x: spatial points
        V: potential as a function of one variable
        k: Bloch wave vector

    Returns:
        ew: sorted eigenvalues (array of length N)
        ef: corresponding eigenvectors, ef[:, i] (size N*N)
    """
    delta_x = x[1] - x[0]
    v_values = V(x=x)                               # potential values
    N = len(x)
    z = hbar_eff**2 / (2.0 * delta_x**2)            # off-diagonal element

    h = (np.diag(v_values + 2.0 * z) +
         np.diag(-z * np.ones(N - 1), k=-1) +
         np.diag(-z * np.ones(N - 1), k=1))         # Hamiltonian matrix

    h = np.complex128(h)
    h[0, -1] = -z * np.exp(-1j * k)
    h[-1, 0] = -z * np.exp(1j * k)

    ew, ef = eigh(h)                                # diagonalization
    ef = ef / np.sqrt(delta_x)                      # normalization
    return ew, ef

def mouse_click(event, ax1, ax2, hbar_eff, x, V, per):
    """
    Plots the eigenfunctions at the level of the eigenvalues in ax2.
    By clicking in ax1, k is selected.
    x is the position and V is the potential.
    """
    mode = event.canvas.toolbar.mode

    # Check if left mouse click inside axes and no active toolbar mode
    if event.button == 1 and event.inaxes == ax1 and mode == '':
        k0 = event.xdata
        ew, ev = diagonalization(hbar_eff, x, V, k0)

        ev_per = np.zeros((per * len(x), len(x)), dtype=np.complex128)
        ew_per = np.zeros(per * len(x))
        title = 'Eigenfunctions for k = {:.2f}'.format(k0)

        # Plot for multiple periods
        for j in range(per):
            ev_per[j * len(x):(j + 1) * len(x), :] = ev * np.exp(1j * k0 * j)
            ew_per[j * len(x):(j + 1) * len(x)] = ew

        x_per = np.linspace(x[0], per * x[-1], per * len(x))

        plot_eigenfunctions(ax2, ew_per, ev_per, x_per, V,
                            width=1, Emax=7, fak=0.6,
                            betragsquadrat=True, basisline=True,
                            alpha=1.0, title=title)

        event.canvas.draw()


        for line in list(ax2.lines): # removes previous eigenfunctions
            line.remove()

def plot_eigenfunctions(ax, ew, ef, x, V, width=1, Emax=7, fak=0.01,
                        betragsquadrat=False, basisline=True,
                        alpha=1.0, title=None):
    """Plot eigenfunctions.

    The lowest eigenfunctions 'ef' in the potential 'V(x)' are plotted
    at the height of their eigenvalues 'ew' in the plot area 'ax2'.

    Optional parameters:
        width: line width
        Emax: maximum energy displayed
        fak: scaling factor for visualization of eigenfunctions
        betragsquadrat: if True, plot |ψ|^2 instead of ψ
        basisline: draw dashed line at each eigenvalue
        alpha: transparency
        title: plot title
    """

    if title is None:
        title = "Asymmetric double well potential"

    v_values = V(x=x)

    ax.autoscale(False)
    ax.axis([0, 4, np.min(v_values), Emax])
    ax.set_xlabel(r'$x$')
    ax.set_title(title)

    ax.plot(x, v_values, linewidth=2, color='0.7')
    num = np.sum(ew <= Emax)

    if basisline:
        for i in range(num):
            ax.plot(x, ew[i] + np.zeros(len(x)), ls='--', color='0.7')

    try:
        iter(width)
    except TypeError:
        width = width * np.ones(num)

    try:
        iter(alpha)
    except TypeError:
        alpha = alpha * np.ones(num)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    if betragsquadrat:
        ax.set_ylabel(r'$|ψ|^2$ at E')
        for i in range(num):
            ax.plot(x, ew[i] + fak * np.abs(ef[:, i])**2,
                    linewidth=width[i],
                    color=colors[i % len(colors)],
                    alpha=alpha[i])
    else:
        ax.set_ylabel(r'$V(x)$, ψ at E')
        for i in range(num):
            ax.plot(x, ew[i] + fak * ef[:, i],
                    linewidth=width[i],
                    color=colors[i % len(colors)],
                    alpha=alpha[i])

def main():
    print(__doc__)

    xmin = 0
    xmax = 1
    heff = 0.2
    A = 0.2
    N = 300
    per = 4

    print('x_min = {}, x_max = {}, effective hbar = {}, A = {}'
          .format(xmin, xmax, heff, A))
    print('Number of discretization points = {}, periods = {}'
          .format(N, per))

    x, dx = discretization(xmin, xmax, N, retstep=True)
    V = functools.partial(periodic_potential, A=A)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.set_ylim(-1, 7)

    length = 6
    k = np.linspace(-np.pi, np.pi, 100)

    k_array = np.zeros([len(k), length])

    for i, l in enumerate(k):
        ew, _ = diagonalization(heff, x, V, l)
        k_array[i, :] = ew[:length]

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for col in range(length):
        ax.plot(k, k_array[:, col], c=colors[col])

    ax.set_xlabel("k")
    ax.set_ylabel("E")
    ax.set_title("Band structure")

    click_function = functools.partial(
        mouse_click, ax1=ax, ax2=ax2,
        x=x, V=V, hbar_eff=heff, per=per
    )

    fig.canvas.mpl_connect('button_press_event', click_function)

    plt.show()

if __name__ == "__main__":
    main()

