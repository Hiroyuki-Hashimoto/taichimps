"""
FixFreeze matching LAMMPS fix freeze.
Zeroes out forces and torques on granular particles to hold them fixed in space.
Reference: LAMMPS src/GRANULAR/fix_freeze.cpp
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixFreeze(Fix):
    def __init__(self, domain: Domain, float_type: Any = ti.f64) -> None:
        super().__init__(domain, float_type)

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        atom: ti.template(),
    ):
        for i in range(nlocal):
            atom.v[i] = ti.Vector([0.0, 0.0, 0.0])
            atom.omega[i] = ti.Vector([0.0, 0.0, 0.0])
            atom.f[i] = ti.Vector([0.0, 0.0, 0.0])
            atom.torque[i] = ti.Vector([0.0, 0.0, 0.0])

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.post_force_kernel(atom.nlocal, atom)
