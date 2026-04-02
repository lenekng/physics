"""
Time evolution of a Gaussian wave packet with mean position x0
and mean momentum p0 in an asymmetric double-well
potential V(x) = x^4 - x^2 - A x"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import functools
import quantenmechanik as qm


def wavepackage(x, x0, delta_x, heff, p0):
    """
    Input:
        x = position array
        x0 = initial position
        delta_x = parameter
        heff = effective reduced Planck constant
        p0 = initial momentum

    Output:
        Gaussian wave packet
    """
    phi = (1/(2 * np.pi * delta_x**2)**(1/4) *
           np.exp(-(x - x0)**2 / (4 * delta_x**2)) *
           np.exp(1j / heff * p0 * x))
    return phi

def harm_Osz(x):
    """
    Potential of the harmonic oscillator. x is the position.
    """
    return 1 / 2 * x**2

def doublewell(x, A):
    """
    Double-well potential. x is the position and A is the asymmetry parameter.
    """
    return x**4 - x**2 - A * x

def Klick(event, ax, x, welle, pot, ew, ev, heff, dx, t_array, skal):
    """
    Computation of the expansion coefficients c_n and of the wave packet Phi_c, and output of the difference.
    By clicking in the plot window, the initial position x0 is set (event); for this position, the time evolution of the wave function in the potential pot is displayed.
    x is the discretized position coordinate, welle is the wave packet with the initial values defined in the main program.
    ax = plot window
    ew = eigenvalues
    ev = eigenvectors
    heff = effective reduced Planck constant
    dx = spatial grid spacing
    t_array = array of time steps
    """
    
    mode = event.canvas.toolbar.mode
    # test if click is the plot and modus is empty
    if event.button == 1 and event.inaxes and mode == '':
        # select start with click
        x0 = event.xdata

        # Computation of the expansion coefficients
        c = dx * np.dot(np.conjugate(np.transpose(ev)), welle(x0 = x0))

        # wavefunction
        phi_c = np.dot(c, ev.T)

        diff = np.linalg.norm(welle(x0 = x0) - phi_c)
        energy = np.dot(np.abs(c)**2, ew)      # energy expectation value

        print("Difference phi - phi_c = {}".format(diff))
        print("energy expectation value = {}\nInitial position = {} "
              .format(energy, x0))

        # start plot
        phi_start = ax.plot(x, energy + np.abs(welle(x0 = x0))**2 * skal)

        # Time evolution of the plot
        for t in t_array:

            phi_t = np.abs(np.dot(c * np.exp(-1j * ew * t / heff), ev.T))**2
            phi_start[0].set_ydata(phi_t * skal + energy) 
            event.canvas.flush_events()
            event.canvas.draw()

    
def main():
    print(__doc__)

    # set parameters
    skal = 0.02
    heff = 0.07
    # Intervall
    xmin = -1.7
    xmax = 1.7

    N = 300         # Number of discretization points
    A = 0.06
    delta_x = 0.1
    p0 = 0.0       # start momentum
    print('Interval: x_min = {}, x_max = {}'.format(xmin, xmax))
    print('Number of discretization points N = {}'.format(N))
    print('h_eff = {}\nA = {}\nScaling = {}'
          .format(heff, A, skal))
    print('Momentum = {}\ndelta_x = {}'
          .format(p0, delta_x))
    
    potential = functools.partial(doublewell, A = A)
    # Spatial discretization
    x, dx = qm.discretization(xmin, xmax, N, retstep=True)

    # calculation of eigenvalues and eigenvectors
    ew, ev = qm.diagonalization(heff, x, potential)

    # time for 0 to 10
    t_array = np.linspace(0.0,10.0,200)

    #  wavepackage
    wave = functools.partial(wavepackage, x = x,
                        delta_x = delta_x, heff = heff, p0 = p0)


    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ew1, ev1 = qm.diagonalization(heff, x, potential)

    # Plot of the eigenfunctions at the level of the eigenenergy
    qm.plot_eigenfunctions(ax, ew1, ev1, x, potential, alpha = 0.5,
                            scale = skal, magnitude_squared=True)
    
    klick_function = functools.partial(Klick, ax = ax, x = x,
                welle = wave, pot = potential,
                ew = ew, ev = ev, heff = heff, dx = dx, t_array = t_array,
                skal = skal)
    
    fig.canvas.mpl_connect("button_press_event", klick_function)
    plt.show()


if __name__ == "__main__":
    main()


