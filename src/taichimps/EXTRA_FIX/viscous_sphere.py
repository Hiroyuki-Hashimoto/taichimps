"""
FixViscousSphere matching LAMMPS fix viscous/sphere (EXTRA-FIX package).
Applies Stokes viscous drag force and torque to finite-size spherical particles:
  F_drag = -6 * pi * mu * r * v
  T_drag = -8 * pi * mu * r^3 * omega
Reference: LAMMPS src/EXTRA-FIX/fix_viscous_sphere.cpp
"""

import math
from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixViscousSphere(Fix):
    def __init__(self, domain: Domain, gamma: float = 1.0, float_type: Any = ti.f64) -> None:
        super().__init__(domain, float_type)
        self.gamma = float(gamma)

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        atom: ti.template(),
    ):
        pi_val = math.pi
        gamma_val = ti.cast(self.gamma, self.float_type)
        for i in range(nlocal):
            r = atom.radius[i]
            # Stokes drag: 6 * pi * mu * r
            drag_lin = 6.0 * pi_val * gamma_val * r
            atom.f[i] -= drag_lin * atom.v[i]
            # Stokes torque: 8 * pi * mu * r^3
            drag_rot = 8.0 * pi_val * gamma_val * (r * r * r)
            atom.torque[i] -= drag_rot * atom.omega[i]

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.post_force_kernel(
            atom.nlocal,
            atom,
        )
