from manim import *
import numpy as np

from src.poisson.no_homogeneo import simulate_poisson_nonhomogeneous


class PoissonNoHomogeneoSeguimiento(Scene):
    def construct(self):
        # Parámetros
        T = 10.0

        # Definimos la intensidad lambda(t)
        def lam_t(t):
            # Misma lambda(t) que en el notebook:
            # lambda(t) = 2 + 3 * sin^2(pi t / T)
            return 2.0 + 3.0 * np.sin(np.pi * t / T) ** 2

        lam_max = 5.0
        rng = np.random.default_rng(seed=42)

        # 1. Simulamos los tiempos de llegada del proceso de Poisson no homogéneo
        arrival_times = simulate_poisson_nonhomogeneous(lam_t, T, lam_max, rng=rng)

        # 2. Construimos el camino escalonado de N(t)
        t_vals = [0.0]
        N_vals = [0]

        N = 0
        for t in arrival_times:
            # Antes del salto, N(t) se mantiene constante
            t_vals.append(t)
            N_vals.append(N)

            # En el instante de llegada, N aumenta en 1
            N += 1
            t_vals.append(t)
            N_vals.append(N)

        # Tras la última llegada, mantener N(t) constante hasta T
        t_vals.append(T)
        N_vals.append(N)

        t_vals = np.array(t_vals)
        N_vals = np.array(N_vals)

        # 3. Creamos los ejes para N(t) (arriba)
        axes_N = Axes(
            x_range=[0, T, 2],
            y_range=[0, max(1, N + 1), 1],
            x_length=10,
            y_length=3,
            axis_config={"include_numbers": True},
        )
        labels_N = axes_N.get_axis_labels(x_label="t", y_label="N(t)")

        title = Text(
            "Proceso de Poisson no homogéneo",
            font_size=28
        ).to_edge(UP)

        # 4. Creamos los ejes para lambda(t) (abajo)
        axes_lam = Axes(
            x_range=[0, T, 2],
            y_range=[0, lam_max + 0.5, 1],
            x_length=10,
            y_length=2,
            axis_config={"include_numbers": True},
        ).next_to(axes_N, DOWN, buff=0.7)

        labels_lam = axes_lam.get_axis_labels(x_label="t", y_label=r"\lambda(t)")

        # 5. Gráfica de lambda(t)
        lam_graph = axes_lam.plot(
            lambda x: lam_t(x),
            x_range=[0, T],
        )

        # 6. Mostramos ejes y títulos
        self.play(Create(axes_N), Write(labels_N))
        self.play(Write(title))
        self.play(Create(axes_lam), Write(labels_lam))
        self.play(Create(lam_graph), run_time=2)

        # 7. ValueTracker para el tiempo "actual"
        t_tracker = ValueTracker(0.0)

        # Punto que se mueve siguiendo lambda(t)
        def get_cursor_dot():
            t = t_tracker.get_value()
            return Dot(
                axes_lam.coords_to_point(t, lam_t(t)),
                color=YELLOW,
                radius=0.06,
            )

        # Línea vertical desde el eje t hasta lambda(t)
        def get_cursor_line():
            t = t_tracker.get_value()
            p_bottom = axes_lam.coords_to_point(t, 0)
            p_top = axes_lam.coords_to_point(t, lam_t(t))
            return Line(
                p_bottom,
                p_top,
                stroke_color=YELLOW,
                stroke_width=2,
            )

        cursor_dot = always_redraw(get_cursor_dot)
        cursor_line = always_redraw(get_cursor_line)

        # Añadimos cursor a la escena
        self.add(cursor_line, cursor_dot)

        # 8. Dibujamos el camino de N(t) mientras el cursor recorre lambda(t)

        # Preparamos los segmentos del camino N(t)
        segments = []
        for i in range(1, len(t_vals)):
            x0, y0 = t_vals[i - 1], N_vals[i - 1]
            x1, y1 = t_vals[i], N_vals[i]
            p0 = axes_N.coords_to_point(x0, y0)
            p1 = axes_N.coords_to_point(x1, y1)
            segments.append(Line(p0, p1))

        last_t = t_vals[0]

        for i, seg in enumerate(segments):
            target_t = t_vals[i + 1]

            # Duración proporcional a la diferencia de tiempo (con mínimo)
            dt = abs(target_t - last_t)
            run_time = max(0.15, dt * 0.4)

            # Animamos simultáneamente:
            # - se dibuja el siguiente tramo de N(t)
            # - el cursor (dot + línea) se mueve sobre lambda(t)
            self.play(
                Create(seg),
                t_tracker.animate.set_value(target_t),
                run_time=run_time,
            )

            last_t = target_t

        # 9. (Opcional) marcar las llegadas en N(t) con puntos
        for k, t in enumerate(arrival_times, start=1):
            p = axes_N.coords_to_point(t, k)
            dot = Dot(p, color=RED, radius=0.04)
            self.play(FadeIn(dot), run_time=0.1)

        self.wait(2)
