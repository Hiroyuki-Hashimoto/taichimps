"""
Velocity Verlet integration for finite-size spherical particles with rotation.
Reference: LAMMPS src/fix_nve_sphere.cpp
License: GPL v2 / taichimps MIT reimplementation

Equations of motion:
  I = 2/5 * m * r^2  (moment of inertia of solid sphere)
  dtv = dt
  dtf = 0.5 * dt

Step 1 (initial_integrate):
  v += dtf * (f / m)
  x += dtv * v
  omega += dtf * (torque / I)
  (PBC wrapping happens at reneighbor time, not here)

Step 2 (force evaluation occurs in pair / gravity / wall)

Step 3 (final_integrate):
  v += dtf * (f / m)
  omega += dtf * (torque / I)
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixNVESphere(Fix):
    """
    Constant NVE integration for spherical particles with translational and rotational degrees of freedom.
    """

    def __init__(self, domain: Domain, float_type: Any = ti.f64) -> None:
        super().__init__(domain, float_type=float_type)
        self.domain: Domain = domain

    @ti.kernel
    def initial_integrate_kernel(
        self,
        nlocal: ti.i32,
        dt: ti.template(),
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
    ):
        dtv = dt
        dtf = 0.5 * dt

        for i in range(nlocal):
            m = rmass[i]
            r = radius[i]
            # Moment of inertia for solid sphere: I = 2/5 * m * r^2
            inertia = 0.4 * m * r * r

            # Update velocity half-step: v(t + dt/2) = v(t) + dtf * (f / m)
            v[i] += dtf * (f[i] / m)

            # Update position full-step: x(t + dt) = x(t) + dtv * v(t + dt/2)
            # No PBC wrap here: LAMMPS remaps coordinates in Domain::pbc(),
            # which only runs on reneighboring steps.  Wrapping every step
            # would make the neighbor list's skin displacement check compare
            # positions from different periodic images.
            x[i] = x[i] + dtv * v[i]

            # Update angular velocity half-step: omega(t + dt/2) = omega(t) + dtf * (torque / I)
            omega[i] += dtf * (torque[i] / inertia)

    @ti.kernel
    def final_integrate_kernel(
        self,
        nlocal: ti.i32,
        dt: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
    ):
        dtf = 0.5 * dt

        for i in range(nlocal):
            m = rmass[i]
            r = radius[i]
            inertia = 0.4 * m * r * r

            # Update velocity 2nd half-step: v(t + dt) = v(t + dt/2) + dtf * (f(t + dt) / m)
            v[i] += dtf * (f[i] / m)

            # Update angular velocity 2nd half-step: omega(t + dt) = omega(t + dt/2) + dtf * (torque(t + dt) / I)
            omega[i] += dtf * (torque[i] / inertia)

    def initial_integrate(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.initial_integrate_kernel(
            atom.nlocal,
            dt,
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
        )

    def final_integrate(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.final_integrate_kernel(
            atom.nlocal,
            dt,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
        )
