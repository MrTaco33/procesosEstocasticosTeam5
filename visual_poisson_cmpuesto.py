from manim import *
import numpy as np

from src.poisson.compuesto import simulate_compound_poisson


class PoissonCompuestoScene(Scene):
    def construct(self):
        # Parámetros de simulación
        lam = 2.0
        T = 5.0
        rng = np.random.default_rng(seed=42)

        # Definimos la distribución de saltos (igual que en el notebook)
        def jump_sampler(rng, size):
            # Saltos exponenciales de media 1
            return rng.exponential(scale=1.0, size=size)

        # Simulamos el proceso compuesto
        arrival_times, X_vals, jumps = simulate_compound_poisson(
            lam=lam,
            T=T,
            jump_sampler=jump_sampler,
            rng=rng
        )

        if len(arrival_times) == 0:
            # Caso raro pero posible: ningún evento en [0, T]
            msg = Text("No hubo saltos en [0, T]", font_size=36)
            self.play(Write(msg))
            self.wait(2)
            return

        # 1. Construimos los puntos (t, X(t)) para la gráfica escalonada
        t_vals = [0.0]
        X_step = [0.0]

        for t, x in zip(arrival_times, X_vals):
            # tramo horizontal antes del salto
            t_vals.append(t)
            X_step.append(X_step[-1])
            # salto: valor después del evento
            t_vals.append(t)
            X_step.append(x)

        t_vals.append(T)
        X_step.append(X_step[-1])

        t_vals = np.array(t_vals)
        X_step = np.array(X_step)

        # Rango en y basado en el valor máximo de X
        y_max = max(X_step[-1] * 1.2, 1.0)

        # 2. Creamos los ejes para X(t)
        axes = Axes(
            x_range=[0, T, 1],
            y_range=[0, y_max, max(0.5, y_max / 6)],
            x_length=10,
            y_length=4,
            axis_config={"include_numbers": True},
        )
        labels = axes.get_axis_labels(x_label="t", y_label="X(t)")

        title = Text(
            "Proceso de Poisson compuesto",
            font_size=28
        ).to_edge(UP)

        self.play(Create(axes), Write(labels))
        self.play(Write(title))
        self.wait(0.5)

        # 3. Dibujamos X(t) segmento a segmento
        segments = []
        for i in range(1, len(t_vals)):
            x0, y0 = t_vals[i - 1], X_step[i - 1]
            x1, y1 = t_vals[i], X_step[i]
            p0 = axes.coords_to_point(x0, y0)
            p1 = axes.coords_to_point(x1, y1)
            segments.append(Line(p0, p1))

        # 4. Vamos animando cada salto, mostrando el tamaño Y_k
        #    Recordemos que:
        #    - arrival_times[k] es el tiempo de llegada k+1
        #    - jumps[k] es Y_{k+1}
        #    - X_vals[k] es X(T_{k+1})

        # Construimos también una lista con (tiempo, índice_k) para llegar a Y_k
        # Los saltos ocurren en los índices impares de t_vals / X_step (por cómo los construimos)
        # pero es más fácil recorrer arrival_times directamente.

        # Mapeamos de tiempo de llegada a índice en t_vals
        # (cada llegada t aparece dos veces seguidas en t_vals)
        jump_index = 0
        last_t = 0.0

        seg_idx = 0  # índice en segments

        for k, (t_jump, y_jump) in enumerate(zip(arrival_times, jumps), start=1):
            # Avanzamos hasta justo antes de este salto (horizontal + vertical)
            # Cada llegada aporta dos segmentos: horizontal hasta t_jump
            # y vertical en t_jump.
            # Para simplificar, dibujamos segmentos de uno en uno.

            # Dibujamos el siguiente par de segmentos (horizontal y vertical)
            for _ in range(2):
                if seg_idx >= len(segments):
                    break
                seg = segments[seg_idx]
                seg_idx += 1
                self.play(Create(seg), run_time=0.2)

            # Marcamos el punto después del salto y el tamaño Y_k
            point_after_jump = axes.coords_to_point(t_jump, X_vals[k - 1])
            dot = Dot(point_after_jump, color=YELLOW, radius=0.05)
            label = MathTex(
                f"Y_{k} = {y_jump:.2f}",
            ).scale(0.6).next_to(dot, UP, buff=0.2)

            self.play(FadeIn(dot), FadeIn(label), run_time=0.3)

        # Dibujamos los segmentos restantes (después del último salto)
        while seg_idx < len(segments):
            seg = segments[seg_idx]
            seg_idx += 1
            self.play(Create(seg), run_time=0.2)

        self.wait(2)
