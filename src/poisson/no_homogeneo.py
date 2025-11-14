# src/poisson/no_homogeneo.py

import numpy as np
from .homogeneo import simulate_poisson_homogeneous


def simulate_poisson_nonhomogeneous(lam_func, T, lam_max, rng=None):
    """
    Simula un proceso de Poisson no homogéneo en el intervalo [0, T]
    usando el método de thinning (adelgazamiento).

    Parámetros
    ----------
    lam_func : callable
        Función lambda(t) que devuelve la intensidad en el tiempo t.
        Debe ser no negativa para todo t en [0, T].
    T : float
        Horizonte de tiempo de simulación (intervalo [0, T]).
    lam_max : float
        Cota superior de lambda(t) en [0, T]. Se requiere que
        lambda(t) <= lam_max para todo t en [0, T].
    rng : np.random.Generator, opcional
        Generador de números aleatorios de NumPy. Si es None,
        se crea uno nuevo.

    Returns
    -------
    arrival_times : np.ndarray
        Arreglo con los tiempos de llegada aceptados (proceso no homogéneo).
    """
    if T <= 0:
        raise ValueError("El horizonte de tiempo T debe ser > 0")
    if lam_max <= 0:
        raise ValueError("lam_max debe ser > 0")

    if rng is None:
        rng = np.random.default_rng()

    # 1. Generamos tiempos candidatos con un Poisson homogéneo de tasa lam_max
    candidate_times = simulate_poisson_homogeneous(lam_max, T, rng=rng)

    accepted_times = []

    # 2. Para cada tiempo candidato, decidimos si se acepta o se rechaza
    for t in candidate_times:
        lam_t = lam_func(t)

        if lam_t < 0:
            raise ValueError("lambda(t) no puede ser negativa")

        if lam_t > lam_max + 1e-8:
            # Pequeña tolerancia numérica, por si hay errores de redondeo
            raise ValueError(
                f"lambda({t}) = {lam_t} excede lam_max = {lam_max}. "
                "Elige un lam_max más grande."
            )

        # Probabilidad de aceptar este evento
        p_accept = lam_t / lam_max

        # Generamos U ~ Uniforme(0,1)
        u = rng.random()

        if u <= p_accept:
            accepted_times.append(t)

    return np.array(accepted_times)
