"""
FixTriaxial: 6-Face Independent Stress / Strain-Rate Servo Controller.

Based on Yade's TriaxialStressController & PeriTriaxController:
Controls confining stress (isotropic consolidation, True Triaxial, plane strain)
or prescribed strain rate loading (axial shearing, cyclic shearing).

Reference:
- Yade TriaxialStressController (Chareyre et al., 2005)
- LAMMPS fix deform / fix press/berendsen
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixTriaxial(Fix):
    """Independent 3-axis stress / strain-rate servo controller for triaxial testing."""

    def __init__(
        self,
        domain: Domain,
        float_type: Any = ti.f64,
        target_stress: Sequence[float] = (-100.0e3, -100.0e3, -100.0e3),
        strain_rate: Sequence[float] = (0.0, 0.0, 0.0),
        stress_mask: Sequence[int] = (1, 1, 1),
        stress_damping: float = 0.5,
        max_velocity: float = 0.1,
    ) -> None:
        super().__init__(domain=domain, float_type=float_type)
        self.target_stress = np.array(target_stress, dtype=np.float64)
        self.strain_rate = np.array(strain_rate, dtype=np.float64)
        self.stress_mask = list(stress_mask)
        self.stress_damping = float(stress_damping)
        self.max_velocity = float(max_velocity)

        # Field for virial stress accumulator
        self.virial = ti.Matrix.field(3, 3, dtype=float_type, shape=())
        self.current_stress = np.zeros(3, dtype=np.float64)

    @ti.kernel
    def compute_virial_stress_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        rmass: ti.template(),
    ):
        self.virial[None] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for i in range(nlocal):
            m = rmass[i]
            # Kinetic part
            for r in ti.static(range(3)):
                for c in ti.static(range(3)):
                    ti.atomic_add(self.virial[None][r, c], m * v[i][r] * v[i][c])
            # Force part
            for r in ti.static(range(3)):
                for c in ti.static(range(3)):
                    ti.atomic_add(self.virial[None][r, c], x[i][r] * f[i][c])

    @ti.kernel
    def remap_positions_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        scale_x: ti.template(),
        scale_y: ti.template(),
        scale_z: ti.template(),
        origin_x: ti.template(),
        origin_y: ti.template(),
        origin_z: ti.template(),
    ):
        for i in range(nlocal):
            x[i][0] = origin_x + (x[i][0] - origin_x) * scale_x
            x[i][1] = origin_y + (x[i][1] - origin_y) * scale_y
            x[i][2] = origin_z + (x[i][2] - origin_z) * scale_z

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0 or self.domain is None:
            return

        vol = self.domain.volume
        if vol <= 0.0:
            return

        # Compute current virial stress
        self.compute_virial_stress_kernel(
            atom.nlocal,
            atom.x,
            atom.v,
            atom.f,
            atom.rmass,
        )
        vir_np = self.virial.to_numpy()
        # Cauchy stress sigma_ii = - virial_ii / volume
        self.current_stress = np.array(
            [-vir_np[0, 0] / vol, -vir_np[1, 1] / vol, -vir_np[2, 2] / vol],
            dtype=np.float64,
        )

        lx = self.domain.boxhi[0] - self.domain.boxlo[0]
        ly = self.domain.boxhi[1] - self.domain.boxlo[1]
        lz = self.domain.boxhi[2] - self.domain.boxlo[2]
        lengths = [lx, ly, lz]

        scale = [1.0, 1.0, 1.0]

        for axis in range(3):
            if self.stress_mask[axis] == 1:
                # Stress servo control (Yade TriaxialStressController formulation)
                delta_sigma = self.current_stress[axis] - self.target_stress[axis]
                # Estimation of bulk/tangent stiffness: K ~ max(|sigma|, 1.0) / L
                k_est = max(abs(self.current_stress[axis]), 1e4) / lengths[axis]
                vel = np.sign(delta_sigma) * min(
                    self.max_velocity,
                    (self.stress_damping * abs(delta_sigma)) / (k_est * dt),
                )
                dL = vel * dt
                scale[axis] = (lengths[axis] + dL) / lengths[axis]
            else:
                # Strain rate control
                dL = self.strain_rate[axis] * lengths[axis] * dt
                scale[axis] = (lengths[axis] + dL) / lengths[axis]

        # Apply boundary expansion / contraction symmetrically
        new_boxlo = [
            self.domain.boxlo[0] - 0.5 * (scale[0] - 1.0) * lx,
            self.domain.boxlo[1] - 0.5 * (scale[1] - 1.0) * ly,
            self.domain.boxlo[2] - 0.5 * (scale[2] - 1.0) * lz,
        ]
        new_boxhi = [
            self.domain.boxhi[0] + 0.5 * (scale[0] - 1.0) * lx,
            self.domain.boxhi[1] + 0.5 * (scale[1] - 1.0) * ly,
            self.domain.boxhi[2] + 0.5 * (scale[2] - 1.0) * lz,
        ]

        # Affine position remap
        self.remap_positions_kernel(
            atom.nlocal,
            atom.x,
            scale[0],
            scale[1],
            scale[2],
            self.domain.boxlo[0],
            self.domain.boxlo[1],
            self.domain.boxlo[2],
        )

        self.domain.set_box(new_boxlo, new_boxhi)


# Alias for LAMMPS style naming
FixTriaxServo = FixTriaxial
