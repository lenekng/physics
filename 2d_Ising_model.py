"""

The left plot shows the the spins (Dimension NxN).
The right plot shows the average magnetization m per spin on the y-axis and the dimensionless temperature on the x-axis.

"""

import numpy as np
import matplotlib.pyplot as plt
import functools
import numpy.random as rnd

class IsingZustand(object):
    '''
    Class to safe the spin and the temperature.
    '''
    def __init__(self, spins, tau):
        '''Initialise: ''spins'' + temperature ''tau''.'''
        self.spins = spins
        self.tau = tau
        
def spinzustand(m, zustand, N):
    """

Generation of the spin state of the class zustand with random,
independent spins whose average magnetization per spin is approximately m.
N × N is the size of the system.
    """
    if m > 1 or m < -1:
        print('m must be between -1 and 1.')
    else:
        p = (np.abs(m) + 1) / 2     # probability
        # new random state matrix whose magnetization is approximately m
        zustand.spins = np.sign(m) * (rnd.binomial(1, p, size = (N,N))*2 - 1)


def monte_carlo(zustand, N, tau):
    """
    Input: zustand = spin state of the system
        N = size of the system (N × N)
        tau = dimensionless time
    The new spin state after one Monte Carlo step is stored in the class.
    """
    for i in range(0, N * N):
        # select a random spin
        x = rnd.randint(0, N)
        y = rnd.randint(0, N)

        # Energy change with periodic boundary conditions
        dE = zustand.spins[x, y]*(zustand.spins[x, (y+1) % N]
                    + zustand.spins[x, (y-1) % N]
                    + zustand.spins[(x+1) % N, y] + zustand.spins[(x-1) % N, y])

        #if dE > 0 the state is flipped with probability  exp(-1/tau * dE)
        if (dE > 0) and (rnd.random() < np.exp(-1 / zustand.tau * dE)):
                zustand.spins[x, y] = zustand.spins[x, y] * (-1)

        # spin flip if dE <= 0
        elif dE <= 0:
                zustand.spins[x, y] = zustand.spins[x, y] * (-1)
                

def maus_klick(event, ax1, ax2, zustand, spins_img, N, Anz):
    """
        Input:
            event      = left mouse click
            ax1, ax2   = axes 1 and 2
            zustand    = state of the class
            spins_img  = instance of imshow
            N          = N x N is the number of spins
            Anz        = number of Monte Carlo steps

        The initial values for m (average magnetization per spin) and tau
        (dimensionless temperature) are selected by clicking in ax2. The
        corresponding spin configurations are drawn in spins_img.

        Clicking in ax1 performs Anz Monte Carlo steps and displays the
        result in ax1.
    """
    mode = event.canvas.toolbar.mode
    # Test, ob Klick mit linker Maustaste in ax2 erfolgt und
    # der Modus leer ist:
    if event.button == 1 and event.inaxes == ax2 and mode == '':
        m0 = event.ydata        # select m
        zustand.tau = event.xdata      # select tau
        spinzustand(m0, zustand, N)     # new spin state
        ax2.plot(zustand.tau, m0, '.', c = 'g')
        
        spins_img.set_data(zustand.spins)
        event.canvas.draw()
        event.canvas.flush_events()
        print('\nnew parameter:')
        print('m = {}\ntau = {}\n'.format(m0, zustand.tau))

    # test, if click is in ax1 Test and modus empty
    if event.button == 1 and event.inaxes == ax1 and mode == '':
        print("It performs {} Monte Carlo steps.".format(Anz))
        
        for i in range(Anz):
            monte_carlo(zustand, N, zustand.tau)
            spins_img.set_data(zustand.spins)

            event.canvas.draw()
            event.canvas.flush_events()
            m = np.sum(zustand.spins) / (N*N)
            ax2.plot(zustand.tau, m, '.', c = 'g')
      

def main():
    print(__doc__)

    # parmeter
    N = 50
    tau = 1.0
    m_start = 0.4
    Anz = 10
    tau_c = 2/np.arcsinh(1)     # critical temperature (dimensionless)

    # new object
    spins = np.ones((N,N))
    ising = IsingZustand(spins, tau)
    spinzustand(m_start, ising, N)

    print("N = {}\ncritical temperature (dimensionless) = {}\n".format(N, tau_c))
    print("Initial parameters:")
    print("Magnetization m0 = {}\ndimensionless temperature tau_0 = {}"
          .format(m_start, tau))

    tau1 = np.linspace(0.1, tau_c, 100, endpoint = False)
    m_end = np.zeros(40)        # nach tau_c m = 0
    tau2 = np.linspace(tau_c, tau_c + 1.5, len(m_end))

    # Analytical prediction according to Yang
    m = (1 - 1 / np.sinh(2 / tau1)**4)**(1/8) 

    m_ges = np.concatenate((m, m_end))
    tau_ges = np.concatenate((tau1, tau2))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax2.set_ylabel("m")
    ax2.set_xlabel(r'$\tau$')
    ax2.set_title('2D Ising Model')
    
    ax1.set_xlabel("x")
    ax1.set_ylabel('y')
    ax1.set_title('Spin field')

    ax2.plot(tau_ges, m_ges, c = 'b')
    ax2.plot(tau_ges, -m_ges, c = 'b')
    plt.axis([0, tau_c + 1.5, -1.1, 1.1])
    ax2.grid()

    vmax = 1
    vmin = -1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    spins_img = ax1.imshow(ising.spins, norm = norm)


    plt.colorbar(spins_img, ax = ax1, shrink=0.4, boundaries = [-1,0,1],
                 values = [-1,1], ticks = [-1,1])
    klick_function = functools.partial(maus_klick, ax2=ax2, ax1=ax1,
                    zustand=ising, spins_img=spins_img, N=N, Anz = Anz)
    fig.canvas.mpl_connect('button_press_event', klick_function)
    plt.show()


if __name__ == "__main__":
    main()
