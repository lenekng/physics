"""
Aufgabe 7.1: Quantenmechanik von 1D Potentialen II
Darstellung der zweitlichen Entwicklung eins Gauß´sches Wellenpaktets mit
mittlerem Ort x0 und mittlerem Impuls p0 in einem asymmetrischen
Doppelmuldenpotential V(x) = x^4 - x^2 - A*x. 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import functools
import quantenmechanik as qm


def wellenpaket(x, x0, delta_x, heff, p0):
    """
    Eingabe:
        x = Ortsarray
        x0 = Startpunkt
        delta_x = Parameter
        heff = effektives hquer
        p0 = Anfangsimpuls

    Ausgabe:
        Gauß´sches Wellenpaket
    """
    phi = (1/(2 * np.pi * delta_x**2)**(1/4) *
           np.exp(-(x - x0)**2 / (4 * delta_x**2)) *
           np.exp(1j / heff * p0 * x))
    return phi

def harm_Osz(x):
    """
    Potential des harmonischer Oszillators. x ist der Ort.
    """
    return 1 / 2 * x**2

def doppelmulde(x, A):
    """
    Doppelmuldenpotential. x ist der Ort und A der Asymmetrisierungsparameter.
    """
    return x**4 - x**2 - A * x

def Klick(event, ax, x, welle, pot, ew, ev, heff, dx, t_array, skal):
    """
    Berechnung der Entwicklungskoeffizienten c_n und des Wellenpakets Phi_c
    und Ausgabe der Differenz. 
    Durch Klick im Plotfenster wird der Startpunkt x0 festgelegt (event),
    für diesen wird die Zeitentwicklung der Wellenfunktion im Potential pot
    dargestellt.
    x ist der diskretisierte Ort, welle das Wellenpaket mit den im
    Hauptprogramm definierten Anfangswerten.
    ax = Plotfenster
    ew = Eigenwerte
    ev = Eigenvektoren
    heff = effektives h_quer
    dx = Ortgitterabstand
    t_array = Array mit Zeitschritten
    """
    
    mode = event.canvas.toolbar.mode
    # Test, ob Klick mit linker Maustaste im Koordinatensystem erfolgt und
    # der Modus leer ist:
    if event.button == 1 and event.inaxes and mode == '':
        # Festlegen des Startpunkts durch Mausklick
        x0 = event.xdata                
        # Berechung Entwicklungskoeffizienten
        c = dx * np.dot(np.conjugate(np.transpose(ev)), welle(x0 = x0))
        # Zusammensetzen der Wellenfunktion aus Koeffizienten und EV
        phi_c = np.dot(c, ev.T)
        #Berechung der Differenz
        diff = np.linalg.norm(welle(x0 = x0) - phi_c)
        energie = np.dot(np.abs(c)**2, ew)      # Energieerwartungswert

        print("Differenz phi - phi_c = {}".format(diff))
        print("Erwartungswert der Energie = {}\nStartpunkt = {} "
              .format(energie, x0))

        # Anfangsplot
        phi_start = ax.plot(x, energie + np.abs(welle(x0 = x0))**2 * skal)

        # Zeitentwicklung des Plots
        for t in t_array:
            # Neue Daten
            phi_t = np.abs(np.dot(c * np.exp(-1j * ew * t / heff), ev.T))**2
            # Plotdaten aktualisieren
            phi_start[0].set_ydata(phi_t * skal + energie) 
            event.canvas.flush_events()             # und dynamisch
            event.canvas.draw()                     # darstellen

    
def main():
    """Hauptprogramm"""
    print(__doc__)

    # Definition Parameter
    skal = 0.02
    heff = 0.07
    # Intervall
    xmin = -1.7
    xmax = 1.7

    N = 300         # Anzahl der Diskretisierungspunkte
    A = 0.06
    delta_x = 0.1
    p0 = 0.0       # Anfangsimpuls
    print('Intervall: x_min = {}, x_max = {}'.format(xmin, xmax))
    print('Anzahl der Diskretisierungspunkte N = {}'.format(N))
    print('h_eff = {}\nA = {}\nSkalierung = {}'
          .format(heff, A, skal))
    print('Impuls = {}\ndelta_x = {}'
          .format(p0, delta_x))
    
    potential = functools.partial(doppelmulde, A = A)
    # Ortsdiskretisierung 
    x, dx = qm.diskretisierung(xmin, xmax, N, retstep=True)

    # Berechung Eigenwerte- und vektoren
    ew, ev = qm.diagonalisierung(heff, x, potential)

    # Zeiten von 0 bis 10
    t_array = np.linspace(0.0,10.0,200)

    # Festlegen der Parameter für das Wellenpaket
    welle = functools.partial(wellenpaket, x = x,
                        delta_x = delta_x, heff = heff, p0 = p0)

    #Festlegen der Parameter für das Potential

    #Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Berechung Eigenwerte, -funktionen
    ew1, ev1 = qm.diagonalisierung(heff, x, potential)

    # Plot der Eigenfkt. auf Höhe der Eigenenergie
    qm.plot_eigenfunktionen(ax, ew1, ev1, x, potential, alpha = 0.5,
                            fak = skal, betragsquadrat=True)
    
    klick_funktion = functools.partial(Klick, ax = ax, x = x,
                welle = welle, pot = potential,
                ew = ew, ev = ev, heff = heff, dx = dx, t_array = t_array,
                skal = skal)
    
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    plt.show()


if __name__ == "__main__":
    main()


"""
a) Wählt man den Startpunkt der Wellenfunktion bei dem Minimum (linke Mulde)
entspricht dies der Nullten Eigenfunktion. Die Bewegung ist gebunden und
entspricht einem effektiven x^2 Potential (harm. Osz.), da das Teilchen
aufgrund der geringen Energie nur die linke, tiefere Mulde sieht. Die Welle
bleibt gaussförmig, ändert jedoch ihr Maximum und die Varianz in zeitlichen
Verlauf deutlich. Der Ort des Maximums varriert leicht innerhalb der Mulde.

Wählt man als Startpunkt ca. bei x = 0, fließt die ursprünglich gaussförmige
Welle gleichmäßig auseinander und osziliert zwischen den Potential. 

b) Wählt man p = 0.3, erhöht sich der Energieerwartungswert der
Eigenfunktionen.
Somit wird die nullte Wellenfkt., welche ihren Startpunkt im Minimum hat,
auch durch die rechte Mulde beeilflusst (sie sieht also beide Mulden).
Durch die höhere Energie erhöht sich die Tunnelfähigkeit, somit kann man
erkennen, dass die Welle auch deutlich sichtbare Aufenteilswsken. in der
rechten Mulde hat.
Die Maxima der Oszilation sind in der linken Mulde jedoch deutlich
größer, als in der rechten. Im Vergleich zu a) ist hier noch der Unterschied,
dass sie die Gaussform während der Zeitentwicklung nicht beibehält.
Die Welle breitet sich zunächst nach rechts aus, da sie einen Anfangsimpuls
in diese Richtung hat.

Wählt man den Startpunkt im Maximum (x = 0), ist der Energieerwartungswert
ebenfalls deutlich größer als für p = 0.
Die Welle oszilliert nun nicht mehr gleichmäßig nach rechts und links, sondern
zunächst deutlich ausgeprägter nach rechts, da sie einen Anfangsimpuls in
diese Richtung hat.
Sie wird dann man dem Potentialwall reflektiert und wandet nach links.

c) Zum Startzeitpunkt ist das Wellenpaket fast aussschließlich in dem
angeklickten Mimimum lokalisiert. Entwickelt es sich zeitlich, nimmt die
Aufentteilswsk. in der anderen Mulde immer weiter zu, so dass irgendwann der
Punkt erreicht ist, wo das Teilchen fast ausschließlich in der anderen Mulde
lokalisiert ist.
Schreitet die Zeit noch weiter fort, wandert es wieder in die ursprüngliche
Mulde usw.
"""
