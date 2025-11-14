def sample_exponential(lam, size=1, rng=None):
    """
    Genera muestras de una distribución Exponencial(lambda)
    usando el método de inversión.

    Parámetros
    ----------
    lam : float
        Parámetro lambda (> 0) de la distribución exponencial.
    size : int, opcional
        Número de muestras a generar.
    rng : np.random.Generator, opcional
        Generador de números aleatorios. Si es None, se crea uno nuevo.

    Returns
    -------
    samples : np.ndarray
        Arreglo de longitud `size` con las muestras generadas.
    """
    if lam <= 0:
        raise ValueError("El parámetro lam debe ser > 0")

    if rng is None:
        rng = np.random.default_rng()

    # Generamos U ~ Uniforme(0,1)
    u = rng.random(size=size)

    # Aplicamos la fórmula X = - (1/lam) * ln(1 - U)
    # Usamos 1-U para evitar problemas si U es muy cercano a 0.
    x = -np.log(1.0 - u) / lam

    return x
