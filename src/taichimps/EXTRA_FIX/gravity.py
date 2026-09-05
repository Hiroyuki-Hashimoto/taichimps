"""
LAMMPS fix gravity implementation in Taichi.
Reference: LAMMPS src/fix_gravity.cpp
License: GPL v2 / taichimps MIT reimplementation

F_gravity = mass * g_vector
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixGravity(Fix):
    """
    Applies uniform gravitational acceleration vector to particles.
    """

    def __init__(
        self,
        domain: Domain,
        magnitude: float,  # e.g. 9.81 m/s^2
        direction: list[float] | tuple[float, float, float] = (0.0, 0.0, -1.0),
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)
        # Normalize direction
        norm = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
        if norm == 0.0:
            norm = 1.0
        gx = magnitude * direction[0] / norm
        gy = magnitude * direction[1] / norm
        gz = magnitude * direction[2] / norm

        self.g = ti.Vector([gx, gy, gz], dt=float_type)

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        f: ti.template(),
        rmass: ti.template(),
    ):
        for i in range(nlocal):
            m = rmass[i]
            f[i] += m * self.g

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.post_force_kernel(atom.nlocal, atom.f, atom.rmass)
