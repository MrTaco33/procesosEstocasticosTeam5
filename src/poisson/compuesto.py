# src/poisson/compuesto.py

import numpy as np
from .homogeneo import simulate_poisson_homogeneous


def simulate_compound_poisson(lam, T, jump_sampler, rng=None):
    """
    Simula un proceso de Poisson compuesto X(t) en el intervalo [0, T].

    N(t) es un proceso de Poisson homogéneo con tasa lam.
    Los tamaños de salto Y_k se generan con jump_sampler.

    Parámetros
    ----------
    lam : float
        Tasa del proceso de Poisson N(t). Debe ser lam > 0.
    T : float
        Horizonte de tiempo de simulación (intervalo [0, T]).
    jump_sampler : callable
        Función que recibe (rng, size) y devuelve un array de longitud `size`
        con muestras independientes de la distribución de los saltos Y_k.
        Ejemplo:
            def jump_sampler(rng, size):
                return rng.exponential(scale=1.0, size=size)
    rng : np.random.Generator, opcional
        Generador de números aleatorios de NumPy. Si es None,
        se crea uno nuevo.

    Returns
    -------
    arrival_times : np.ndarray
        Tiempos de llegada T1, T2, ..., Tn del proceso N(t) (todos <= T).
    X_vals : np.ndarray
        Valores del proceso X(t) justo después de cada llegada,
        es decir X(T_k) = sum_{i=1}^k Y_i.
    jumps : np.ndarray
        Tamaños de salto Y_1, ..., Y_n.
    """
    if lam <= 0:
        raise ValueError("La tasa lam debe ser > 0")
    if T <= 0:
        raise ValueError("El horizonte T debe ser > 0")

    if rng is None:
        rng = np.random.default_rng()

    # 1. Simulamos los tiempos de llegada del Poisson homogéneo
    arrival_times = simulate_poisson_homogeneous(lam, T, rng=rng)

    n_jumps = len(arrival_times)
    if n_jumps == 0:
        # No hubo eventos en [0, T]
        return arrival_times, np.array([]), np.array([])

    # 2. Simulamos los tamaños de salto Y_k
    jumps = jump_sampler(rng, n_jumps)

    # 3. Construimos la suma acumulada: X(T_k) = sum_{i=1}^k Y_i
    X_vals = np.cumsum(jumps)

    return arrival_times, X_vals, jumps
