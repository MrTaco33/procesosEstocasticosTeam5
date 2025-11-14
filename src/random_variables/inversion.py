import numpy as np

def sample_discrete(values, probs, size=1, rng=None):
    """
    Genera muestras de una variable aleatoria discreta usando
    el método de inversión de la función de distribución acumulada.

    Parámetros
    ----------
    values : array-like
        Valores posibles de la variable aleatoria (x_1, ..., x_n).
    probs : array-like
        Probabilidades asociadas (p_1, ..., p_n). No es necesario
        que sumen exactamente 1; se normalizan internamente.
    size : int, opcional
        Número de muestras a generar.
    rng : np.random.Generator, opcional
        Generador de números aleatorios de NumPy. Si es None, se crea uno nuevo.

    Returns
    -------
    samples : np.ndarray
        Arreglo de longitud `size` con las muestras generadas.
    """
    values = np.asarray(values)
    probs = np.asarray(probs, dtype=float)

    # Normalizamos por si no suman exactamente 1 (pequeños errores numéricos)
    probs = probs / probs.sum()

    # Calculamos la acumulada
    cum_probs = np.cumsum(probs)

    if rng is None:
        rng = np.random.default_rng()

    # Generamos uniformes en [0, 1)
    u = rng.random(size=size)

    # searchsorted encuentra el índice donde se insertaría cada u
    # para mantener el orden en cum_probs
    indices = np.searchsorted(cum_probs, u)

    return values[indices]
