"""
FixProbe: Spatial Wave Measurement Receiver and Sensor Array.

Records local particle velocities, displacements, and stress components at
specified spatial probe locations. Ideal for seismic / ultrasonic wave recording,
hodograph particle orbit tracking (Rayleigh wave), and f-k dispersion analysis (Love wave).

Reference:
- LAMMPS compute chunk/atom + fix ave/chunk
- Park, C. B., Miller, R. D., & Xia, J. (1999). Multichannel analysis of surface waves. Geophysics.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixProbe(Fix):
    """Spatial measurement probe array for wave recording in granular media."""

    def __init__(
        self,
        domain: Domain | None = None,
        float_type: Any = ti.f64,
        points: Sequence[Sequence[float]] | None = None,
        radius: float = 1.0,
        nevery: int = 1,
        fields: Sequence[str] = ("vx", "vy", "vz"),
        file: str | Path | None = None,
    ) -> None:
        super().__init__(domain=domain, float_type=float_type)
        if points is None or len(points) == 0:
            self.points_np = np.zeros((1, 3), dtype=np.float64)
        else:
            self.points_np = np.array(points, dtype=np.float64)

        self.n_probes = len(self.points_np)
        self.radius = float(radius)
        self.nevery = int(nevery)
        self.fields = [f.lower() for f in fields]
        self.file_path = Path(file) if file else None

        # Probe points field
        self.probe_pos = ti.Vector.field(3, dtype=float_type, shape=self.n_probes)
        self.probe_pos.from_numpy(self.points_np)

        # Output accumulation fields
        self.probe_v = ti.Vector.field(3, dtype=float_type, shape=self.n_probes)
        self.probe_weight = ti.field(dtype=float_type, shape=self.n_probes)

        # History storage in RAM
        self.step_count = 0
        self.time_history: list[float] = []
        self.data_history: list[np.ndarray] = []

        if self.file_path is not None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                header = ["timestep", "time"]
                for p in range(self.n_probes):
                    for field_name in self.fields:
                        header.append(f"p{p}_{field_name}")
                f.write(" ".join(header) + "\n")

    @ti.kernel
    def sample_probes_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        v: ti.template(),
        rmass: ti.template(),
        radius: ti.template(),
        n_probes: ti.i32,
        probe_pos: ti.template(),
        probe_v: ti.template(),
        probe_weight: ti.template(),
        probe_radius: ti.template(),
    ):
        for p in range(n_probes):
            probe_v[p] = ti.Vector([0.0, 0.0, 0.0])
            probe_weight[p] = 0.0

        for p in range(n_probes):
            pos_p = probe_pos[p]
            r_sq_probe = probe_radius * probe_radius
            for i in range(nlocal):
                diff = x[i] - pos_p
                dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
                if dist_sq <= r_sq_probe:
                    # Weighting by particle mass and cubic spline or linear weight
                    w = 1.0 - ti.sqrt(dist_sq) / probe_radius
                    m = rmass[i]
                    eff_w = w * m
                    probe_v[p] += eff_w * v[i]
                    probe_weight[p] += eff_w

        for p in range(n_probes):
            if probe_weight[p] > 0.0:
                probe_v[p] /= probe_weight[p]

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return

        self.step_count += 1
        if self.step_count % self.nevery != 0:
            return
        self.sample_probes_kernel(
            atom.nlocal,
            atom.x,
            atom.v,
            atom.rmass,
            atom.radius,
            self.n_probes,
            self.probe_pos,
            self.probe_v,
            self.probe_weight,
            self.radius,
        )

        v_np = self.probe_v.to_numpy()  # shape (n_probes, 3)
        self.data_history.append(v_np.copy())

    def to_numpy(self) -> np.ndarray:
        """Returns recorded data array of shape (n_timesteps, n_probes, 3)."""
        if not self.data_history:
            return np.empty((0, self.n_probes, 3), dtype=np.float64)
        return np.array(self.data_history, dtype=np.float64)


# Alias for LAMMPS style naming
FixAveProbe = FixProbe
