"""
FixWave: Dynamic Wave Generator and Source Injector.

Generates P-waves (compressional), S-waves (shear), Rayleigh/Love waves
using Ricker wavelets, sinusoidal bursts, or impulse excitations.

Reference:
- LAMMPS fix addforce & fix move
- Ricker, N. (1953). The form and laws of propagation of seismic wavelets. Geophysics.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixWave(Fix):
    """Wave excitation injector for granular media.

    Applies dynamic forces or prescribed velocities to a region or group
    of particles to launch P, S, or surface waves.
    """

    def __init__(
        self,
        domain: Domain | None = None,
        float_type: Any = ti.f64,
        axis: str = "z",
        direction: Sequence[float] | None = None,
        waveform: str = "ricker",
        f0: float = 100.0,
        t0: float | None = None,
        amplitude: float = 1.0,
        mode: str = "force",
        region: Sequence[float] | None = None,
    ) -> None:
        super().__init__(domain=domain, float_type=float_type)
        self.axis_name = axis.lower()
        if direction is not None:
            norm = np.linalg.norm(direction)
            self.dir = [d / norm for d in direction] if norm > 0 else [0.0, 0.0, 1.0]
        else:
            if self.axis_name == "x":
                self.dir = [1.0, 0.0, 0.0]
            elif self.axis_name == "y":
                self.dir = [0.0, 1.0, 0.0]
            else:
                self.dir = [0.0, 0.0, 1.0]

        self.waveform = waveform.lower()
        self.f0 = float(f0)
        self.t0 = float(t0) if t0 is not None else (1.0 / self.f0 if self.f0 > 0 else 0.0)
        self.amplitude = float(amplitude)
        self.mode = mode.lower()
        self.region = list(region) if region is not None else [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9]
        self.current_time = 0.0

    def compute_wave_val(self, t: float) -> float:
        """Compute the scalar amplitude of the waveform at time t."""
        val = 0.0
        if self.waveform == "ricker":
            tau = np.pi * self.f0 * (t - self.t0)
            val = self.amplitude * (1.0 - 2.0 * tau * tau) * np.exp(-tau * tau)
        elif self.waveform == "sine":
            val = self.amplitude * np.sin(2.0 * np.pi * self.f0 * t)
        elif self.waveform == "pulse":
            period = 1.0 / self.f0 if self.f0 > 0 else 1.0
            if 0.0 <= t <= period:
                val = self.amplitude * np.sin(np.pi * self.f0 * t)
        elif self.waveform == "gaussian":
            sigma = 1.0 / (2.0 * np.pi * self.f0) if self.f0 > 0 else 1.0
            diff = t - self.t0
            val = self.amplitude * np.exp(-0.5 * (diff / sigma) ** 2)
        else:
            val = self.amplitude
        return float(val)

    @ti.kernel
    def apply_force_kernel(
        self,
        nlocal: ti.i32,
        atom: ti.template(),
        val_x: ti.template(),
        val_y: ti.template(),
        val_z: ti.template(),
        xlo: ti.template(),
        ylo: ti.template(),
        zlo: ti.template(),
        xhi: ti.template(),
        yhi: ti.template(),
        zhi: ti.template(),
    ):
        for i in range(nlocal):
            px = atom.x[i][0]
            py = atom.x[i][1]
            pz = atom.x[i][2]
            if xlo <= px <= xhi and ylo <= py <= yhi and zlo <= pz <= zhi:
                atom.f[i][0] += val_x
                atom.f[i][1] += val_y
                atom.f[i][2] += val_z

    @ti.kernel
    def apply_velocity_kernel(
        self,
        nlocal: ti.i32,
        atom: ti.template(),
        val_x: ti.template(),
        val_y: ti.template(),
        val_z: ti.template(),
        xlo: ti.template(),
        ylo: ti.template(),
        zlo: ti.template(),
        xhi: ti.template(),
        yhi: ti.template(),
        zhi: ti.template(),
    ):
        for i in range(nlocal):
            px = atom.x[i][0]
            py = atom.x[i][1]
            pz = atom.x[i][2]
            if xlo <= px <= xhi and ylo <= py <= yhi and zlo <= pz <= zhi:
                atom.v[i][0] = val_x
                atom.v[i][1] = val_y
                atom.v[i][2] = val_z

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        w_val = self.compute_wave_val(self.current_time + dt)
        fx = w_val * self.dir[0]
        fy = w_val * self.dir[1]
        fz = w_val * self.dir[2]

        if self.mode == "force":
            self.apply_force_kernel(
                atom.nlocal,
                atom,
                fx,
                fy,
                fz,
                self.region[0],
                self.region[1],
                self.region[2],
                self.region[3],
                self.region[4],
                self.region[5],
            )
        elif self.mode == "velocity":
            self.apply_velocity_kernel(
                atom.nlocal,
                atom,
                fx,
                fy,
                fz,
                self.region[0],
                self.region[1],
                self.region[2],
                self.region[3],
                self.region[4],
                self.region[5],
            )

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        self.current_time += dt
