"""
Real-time 3D Visualizer for taichimps using Taichi GGUI (ti.ui).
Zero-overhead, isolated visualization that does not affect LAMMPS core logic.
"""

from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain


@ti.data_oriented
class Visualizer3D:
    """
    Real-time 3D Particle Visualizer using Taichi GGUI.
    GPU zero-copy rendering directly from AtomSystem field.
    """

    def __init__(
        self,
        domain: Domain,
        max_particles: int = 200000,
        res: tuple[int, int] = (1280, 720),
        title: str = "taichimps Real-time DEM Viewer",
        show_window: bool = True,
    ) -> None:
        self.domain = domain
        self.max_particles = max_particles
        self.res = res
        self.title = title
        self.show_window = show_window

        # GPU rendering buffers (single precision f32 for Vulkan GGUI pipeline)
        self.render_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.render_color = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.render_radius = ti.field(dtype=ti.f32, shape=max_particles)

        # Initialize GGUI window & scene
        self.window = ti.ui.Window(
            self.title,
            self.res,
            show_window=self.show_window,
            vsync=True,
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()

        # Set default camera position based on domain bounds
        box_center = [
            0.5 * (self.domain.boxlo[d] + self.domain.boxhi[d]) for d in range(3)
        ]
        box_diag = max(
            self.domain.boxhi[d] - self.domain.boxlo[d] for d in range(3)
        )
        self.camera.position(
            box_center[0],
            box_center[1] - 2.2 * box_diag,
            box_center[2] + 1.2 * box_diag,
        )
        self.camera.lookat(box_center[0], box_center[1], box_center[2])
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.fov(55)

        self.paused = False

    @ti.kernel
    def _update_render_buffers(
        self,
        nlocal: ti.i32,
        atom: ti.template(),
        color_by_speed: ti.i32,
        max_speed: ti.f32,
    ):
        for i in range(nlocal):
            self.render_pos[i] = ti.cast(atom.x[i], ti.f32)
            self.render_radius[i] = ti.cast(atom.radius[i], ti.f32)

            if color_by_speed == 1:
                vel = ti.cast(atom.v[i], ti.f32)
                speed = vel.norm()
                norm_speed = ti.min(1.0, speed / (max_speed + 1e-6))

                # Jet colormap approximation (Blue -> Cyan -> Yellow -> Red)
                r = ti.cast(ti.min(1.0, ti.max(0.0, 1.5 - ti.abs(norm_speed * 4.0 - 3.0))), ti.f32)
                g = ti.cast(ti.min(1.0, ti.max(0.0, 1.5 - ti.abs(norm_speed * 4.0 - 2.0))), ti.f32)
                b = ti.cast(ti.min(1.0, ti.max(0.0, 1.5 - ti.abs(norm_speed * 4.0 - 1.0))), ti.f32)
                self.render_color[i] = ti.Vector([r, g, b])
            else:
                # Default gold particle color
                self.render_color[i] = ti.Vector([ti.f32(0.9), ti.f32(0.7), ti.f32(0.2)])

    def render_frame(
        self,
        atom: AtomSystem,
        particle_radius: float | None = None,
        color_by: str = "speed",
        max_speed: float = 1.0,
        ui_callback: Any = None,
    ) -> bool:
        """
        Renders current particle configuration to the window.
        Returns False if user requested window close, True otherwise.
        """
        if not self.window.running:
            return False

        # Handle keyboard interactions
        if self.show_window:
            if self.window.get_event(ti.ui.PRESS):
                if self.window.event.key == ti.ui.ESCAPE:
                    self.window.destroy()
                    return False
                elif self.window.event.key == ti.ui.SPACE:
                    self.paused = not self.paused

            # Update camera controls
            self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        # Lighting
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(1.0, 1.0, 3.0), color=(0.8, 0.8, 0.8))

        # Transfer positions & colors directly on GPU
        color_flag = 1 if color_by == "speed" else 0
        n_render = min(atom.nlocal, self.max_particles)
        if n_render > 0:
            self._update_render_buffers(
                n_render,
                atom,
                color_flag,
                float(max_speed),
            )

            # Draw 3D instanced particles
            r_scale = particle_radius if particle_radius is not None else float(np.mean(atom.radius.to_numpy()[:n_render]))
            self.scene.particles(
                self.render_pos,
                radius=r_scale,
                per_vertex_color=self.render_color,
                index_count=n_render,
            )

        self.canvas.scene(self.scene)
        if self.show_window:
            if ui_callback is not None:
                gui = self.window.get_gui()
                ui_callback(gui)
            self.window.show()
        return True
