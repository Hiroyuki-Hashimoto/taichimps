"""
Cundall Non-viscous Local Damping Fix.
Reference: LAMMPS src/GRANULAR/fix_damping_cundall.cpp
License: GPL v2 compatible / MIT reimplementation
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixDampingCundall(Fix):
    """
    Applies non-viscous damping (Cundall damping) to linear forces and angular torques.
    f_i = f_i - gamma_lin * sign(f_i * v_i) * abs(f_i)
    t_i = t_i - gamma_ang * sign(t_i * omega_i) * abs(t_i)
    """

    def __init__(
        self,
        domain: Domain | None = None,
        gamma_lin: float = 0.3,
        gamma_ang: float = 0.3,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type)
        self.gamma_lin = float(gamma_lin)
        self.gamma_ang = float(gamma_ang)

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        atom: ti.template(),
        gamma_l: ti.template(),
        gamma_a: ti.template(),
    ):
        for i in range(nlocal):
            for d in ti.static(range(3)):
                # Linear damping
                work_lin = atom.f[i][d] * atom.v[i][d]
                sign_f = 1.0 if work_lin >= 0.0 else -1.0
                atom.f[i][d] *= 1.0 - gamma_l * sign_f

                # Angular damping
                work_ang = atom.torque[i][d] * atom.omega[i][d]
                sign_t = 1.0 if work_ang >= 0.0 else -1.0
                atom.torque[i][d] *= 1.0 - gamma_a * sign_t

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.post_force_kernel(
            atom.nlocal,
            atom,
            self.gamma_lin,
            self.gamma_ang,
        )
