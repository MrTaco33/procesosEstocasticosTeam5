from manim import *
import numpy as np

# Importamos nuestra función de simulación del proceso de Poisson homogéneo
from src.poisson.homogeneo import simulate_poisson_homogeneous


class PoissonHomogeneoScene(Scene):
    def construct(self):
        # Parámetros del proceso
        lam = 3.0   # tasa λ
        T = 5.0     # horizonte de tiempo
        rng = np.random.default_rng(seed=42)

        # 1. Simulamos los tiempos de llegada del proceso de Poisson
        arrival_times = simulate_poisson_homogeneous(lam, T, rng=rng)

        # 2. Construimos los puntos (t, N(t)) para un gráfico escalonado
        #    Vamos a generar los puntos para la función N(t) pieza a pieza.
        t_vals = [0.0]
        N_vals = [0]

        N = 0  # contador de eventos
        for t in arrival_times:
            # Antes del salto, N(t) se mantiene constante
            t_vals.append(t)
            N_vals.append(N)

            # En el instante de llegada, N salta en +1
            N += 1
            t_vals.append(t)
            N_vals.append(N)

        # Después de la última llegada, mantenemos N(t) constante hasta T
        t_vals.append(T)
        N_vals.append(N)

        t_vals = np.array(t_vals)
        N_vals = np.array(N_vals)

        # 3. Creamos los ejes para dibujar N(t)
        axes = Axes(
            x_range=[0, T, 1],       # rango en el eje x: [0, T]
            y_range=[0, max(1, N+1), 1],  # rango en el eje y
            x_length=10,
            y_length=4,
            axis_config={"include_numbers": True},
        )
        labels = axes.get_axis_labels(x_label="t", y_label="N(t)")

        title = Text(
            f"Proceso de Poisson homogéneo (λ = {lam})",
            font_size=28
        ).to_edge(UP)

        # 4. Animamos la aparición de los ejes y el título
        self.play(Create(axes), Write(labels))
        self.play(Write(title))
        self.wait(0.5)

        # 5. Dibujamos el camino escalonado de N(t) segmento a segmento
        prev_point = axes.coords_to_point(t_vals[0], N_vals[0])

        # Vamos a almacenar los segmentos para mostrarlos poco a poco
        for i in range(1, len(t_vals)):
            # Coordenadas en (t, N(t))
            x = t_vals[i]
            y = N_vals[i]
            curr_point = axes.coords_to_point(x, y)

            # Creamos un segmento entre el punto anterior y el actual
            segment = Line(prev_point, curr_point)

            # Animamos la creación del segmento
            self.play(Create(segment), run_time=0.25)
            prev_point = curr_point

        # 6. Opcional: marcamos las llegadas con puntos y pequeños textos
        for k, t in enumerate(arrival_times, start=1):
            # punto en (t, N(t)) justo después del salto
            point = axes.coords_to_point(t, k)
            dot = Dot(point)
            label = MathTex(f"T_{k}").scale(0.6).next_to(dot, UP, buff=0.1)
            self.play(FadeIn(dot), FadeIn(label), run_time=0.2)

        self.wait(2)