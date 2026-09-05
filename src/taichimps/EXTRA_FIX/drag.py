"""
FixDrag matching LAMMPS fix drag.
Applies velocity-opposing drag force towards coordinates:
  F = -f_drag * v
Reference: LAMMPS src/fix_drag.cpp
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixDrag(Fix):
    def __init__(self, domain: Domain, f_drag: float = 1.0, float_type: Any = ti.f64) -> None:
        super().__init__(domain, float_type)
        self.f_drag = float(f_drag)

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        v: ti.template(),
        f: ti.template(),
    ):
        f_val = ti.cast(self.f_drag, self.float_type)
        for i in range(nlocal):
            f[i] -= f_val * v[i]

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.post_force_kernel(atom.nlocal, atom.v, atom.f)
