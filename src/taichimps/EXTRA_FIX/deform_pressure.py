"""
Deform Pressure Servo Fix for Isotropic and Triaxial Compression.
Reference: LAMMPS src/EXTRA-FIX/fix_deform_pressure.cpp
License: GPL v2 compatible / MIT reimplementation
"""

from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.computes.thermo import Computes
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixDeformPressure(Fix):
    """
    Dynamically adjusts periodic box size (and remaps particle positions)
    to achieve a target confining pressure.
    h_rate = pgain * (p_current - p_target)
    """

    def __init__(
        self,
        domain: Domain,
        p_target: float = 50000.0,
        p_gain: float = 0.0001,
        max_rate: float = 0.1,
        nevery: int = 1,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type)
        self.p_target = float(p_target)
        self.p_gain = float(p_gain)
        self.max_rate = float(max_rate)
        self.nevery = int(nevery)
        self.step_count = 0
        self.computes = Computes(float_type=float_type)

    @ti.kernel
    def remap_positions(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        scale_x: ti.template(),
        scale_y: ti.template(),
        scale_z: ti.template(),
        boxlo_x: ti.template(),
        boxlo_y: ti.template(),
        boxlo_z: ti.template(),
    ):
        for i in range(nlocal):
            x[i][0] = boxlo_x + (x[i][0] - boxlo_x) * scale_x
            x[i][1] = boxlo_y + (x[i][1] - boxlo_y) * scale_y
            x[i][2] = boxlo_z + (x[i][2] - boxlo_z) * scale_z

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        self.step_count += 1
        if self.step_count % self.nevery != 0 or self.domain is None:
            return

        # Current virial / kinetic pressure
        pressures = self.computes.compute_pressure_tensor(atom, self.domain)
        # Average hydrostatic pressure: (pxx + pyy + pzz) / 3
        p_current = float(np.mean(pressures[:3]))

        # Calculate rate of strain for box deformation
        # In LAMMPS deform/pressure:
        # positive difference (p_current > p_target) -> box should expand -> h_rate > 0
        h_rate = self.p_gain * (p_current - self.p_target)
        if abs(h_rate) > self.max_rate:
            h_rate = np.sign(h_rate) * self.max_rate

        # Update box dimensions
        # shift = prd * dt * h_rate
        # boxlo decreases by shift/2, boxhi increases by shift/2
        current_lx = self.domain.boxhi[0] - self.domain.boxlo[0]
        current_ly = self.domain.boxhi[1] - self.domain.boxlo[1]
        current_lz = self.domain.boxhi[2] - self.domain.boxlo[2]

        shift_x = current_lx * dt * h_rate
        shift_y = current_ly * dt * h_rate
        shift_z = current_lz * dt * h_rate

        new_lx = current_lx + shift_x
        new_ly = current_ly + shift_y
        new_lz = current_lz + shift_z

        scale_x = new_lx / current_lx
        scale_y = new_ly / current_ly
        scale_z = new_lz / current_lz

        new_boxlo = [
            self.domain.boxlo[0] - 0.5 * shift_x,
            self.domain.boxlo[1] - 0.5 * shift_y,
            self.domain.boxlo[2] - 0.5 * shift_z,
        ]
        new_boxhi = [
            self.domain.boxhi[0] + 0.5 * shift_x,
            self.domain.boxhi[1] + 0.5 * shift_y,
            self.domain.boxhi[2] + 0.5 * shift_z,
        ]

        # Remap particles affine
        self.remap_positions(
            atom.nlocal,
            atom.x,
            scale_x,
            scale_y,
            scale_z,
            self.domain.boxlo[0],
            self.domain.boxlo[1],
            self.domain.boxlo[2],
        )

        # Apply updated box boundaries
        self.domain.set_box(new_boxlo, new_boxhi)
