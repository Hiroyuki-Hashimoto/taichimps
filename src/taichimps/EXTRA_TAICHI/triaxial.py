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
from taichimps.computes.thermo import Computes
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

        self.computes = Computes(float_type=float_type)
        # Scratch for the remap kernel, held as fields so the kernel compiles
        # once rather than once per distinct box.
        self._old_lo = ti.Vector.field(3, dtype=float_type, shape=())
        self._new_lo = ti.Vector.field(3, dtype=float_type, shape=())
        self._scale = ti.Vector.field(3, dtype=float_type, shape=())
        self.current_stress = np.zeros(3, dtype=np.float64)

    @ti.kernel
    def remap_positions_kernel(self, nlocal: ti.i32, x: ti.template()):
        """
        Affine remap onto the new box: x_new = new_lo + (x - old_lo) * scale.

        Anchoring on the *new* lower bound is what keeps fractional coordinates
        fixed. Anchoring on the old one while the box is recentred (which is
        what this did) leaves every particle displaced by half the box-length
        change on each update, so the assembly drifts out of the cell.
        """
        old_lo = self._old_lo[None]
        new_lo = self._new_lo[None]
        scale = self._scale[None]
        for i in range(nlocal):
            for d in ti.static(range(3)):
                x[i][d] = new_lo[d] + (x[i][d] - old_lo[d]) * scale[d]

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0 or self.domain is None:
            return

        vol = self.domain.volume
        if vol <= 0.0:
            return

        # Current Cauchy stress, sigma = -P. The pressure comes from the
        # pairwise virial tallied by the pair styles rather than from
        # sum(x_i . f_i), which is not translation invariant under PBC.
        pressure = self.computes.compute_pressure_tensor(atom, self.domain)
        self.current_stress = -pressure[:3].copy()

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

        # Affine position remap, anchored on the new lower bound
        self._old_lo[None] = ti.Vector(
            [float(self.domain.boxlo[d]) for d in range(3)]
        )
        self._new_lo[None] = ti.Vector([float(v) for v in new_boxlo])
        self._scale[None] = ti.Vector([float(s) for s in scale])
        self.remap_positions_kernel(atom.nlocal, atom.x)

        self.domain.set_box(new_boxlo, new_boxhi)


# Alias for LAMMPS style naming
FixTriaxServo = FixTriaxial
