# src/poisson/homogeneo.py

import numpy as np


def simulate_poisson_homogeneous(lam, T, rng=None):
    """
    Simula un proceso de Poisson homogéneo con tasa lam en el intervalo [0, T].

    Parámetros
    ----------
    lam : float
        Tasa (lambda > 0) del proceso de Poisson.
    T : float
        Tiempo final de simulación. Simulamos en el intervalo [0, T].
    rng : np.random.Generator, opcional
        Generador de números aleatorios. Si es None, se crea uno nuevo.

    Returns
    -------
    arrival_times : np.ndarray
        Arreglo con los tiempos de llegada T1, T2, ..., Tk (todos <= T).
    """
    if lam <= 0:
        raise ValueError("La tasa lam debe ser > 0")
    if T <= 0:
        raise ValueError("El horizonte de tiempo T debe ser > 0")

    if rng is None:
        rng = np.random.default_rng()

    t = 0.0  # tiempo actual
    arrival_times = []  # aquí vamos guardando las llegadas

    # Mientras no nos pasemos del horizonte
    while True:
        # Generamos un tiempo entre llegadas S ~ Exp(lam)
        # np.random.exponential tiene escala = 1/lam
        s = rng.exponential(scale=1.0 / lam)

        t = t + s  # avanzamos el tiempo

        if t > T:
            # Si nos pasamos de T, no contamos esta llegada y paramos
            break

        arrival_times.append(t)

    return np.array(arrival_times)
