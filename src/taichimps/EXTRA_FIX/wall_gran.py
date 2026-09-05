"""
LAMMPS fix wall/gran implementation in Taichi.
Reference: LAMMPS src/GRANULAR/fix_wall_gran.cpp
License: GPL v2 / taichimps MIT reimplementation

Supports flat planar boundaries in x, y, or z (lo or hi).
Applies normal and tangential Hookean contact force between particle and planar wall.
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixWallGran(Fix):
    """
    Planar granular wall interaction.
    wall_axis: 0 for x, 1 for y, 2 for z
    wall_side: -1 for lower wall (plane normal points +axis), +1 for upper wall (plane normal points -axis)
    wall_coord: coordinate position of plane along wall_axis
    """

    def __init__(
        self,
        domain: Domain,
        wall_axis: int,  # 0, 1, or 2
        wall_side: int,  # -1 (lower boundary) or +1 (upper boundary)
        wall_coord: float,
        kn: float,
        gamman: float,
        kt: float,
        gammat: float,
        xmu: float,
        dampflag: int = 1,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)
        self.wall_axis = wall_axis
        self.wall_side = wall_side
        self.wall_coord = float(wall_coord)
        self.kn = float(kn)
        self.gamman = float(gamman)
        self.kt = float(kt)
        self.gammat = float(gammat)
        self.xmu = float(xmu)
        self.dampflag = dampflag

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        dt: ti.template(),
        wall_axis: ti.i32,
        wall_side: ti.i32,
        wall_coord: ti.template(),
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
    ):
        # Normal pointing into the domain (away from wall):
        # If wall_side == -1 (lower wall at z_lo), wall is below particles, normal is +axis: [0, 0, 1]
        # If wall_side == 1 (upper wall at z_hi), wall is above particles, normal is -axis: [0, 0, -1]
        n_wall = ti.Vector([0.0, 0.0, 0.0])
        if wall_side == -1:
            n_wall[wall_axis] = 1.0
        else:
            n_wall[wall_axis] = -1.0

        for i in range(nlocal):
            pos_axis = x[i][wall_axis]
            r_i = radius[i]

            # Signed distance from wall plane to particle center in direction of normal
            # For lower wall (n = +1): dist = pos - coord. Overlap = r_i - dist
            # For upper wall (n = -1): dist = coord - pos. Overlap = r_i - dist
            dist = 0.0
            if wall_side == -1:
                dist = pos_axis - wall_coord
            else:
                dist = wall_coord - pos_axis

            delta = r_i - dist
            if delta > 0.0:
                mi = rmass[i]
                vi = v[i]
                oi = omega[i]

                # Relative velocity at wall contact:
                # Particle contact point: x[i] - r_i * n_wall
                # Surface velocity of particle at contact: vi - r_i * (oi x n_wall)
                # Wall is stationary: v_wall = 0
                vr = vi - r_i * oi.cross(n_wall)
                vn = vr.dot(n_wall)
                vrn = vn * n_wall
                vrt = vr - vrn

                # Normal contact force
                fn_elastic = self.kn * delta
                fn_damp = self.gamman * vn
                if self.dampflag == 1:
                    fn_damp *= mi

                fn = fn_elastic - fn_damp
                fn = max(fn, 0.0)

                # Tangential contact force (damping / friction)
                ft_damp = self.gammat * vrt
                if self.dampflag == 1:
                    ft_damp *= mi

                ft_vec = -ft_damp
                ft_mag = ft_vec.norm()
                ft_max = self.xmu * fn_elastic

                if ft_mag > ft_max and ft_mag > 1e-16:
                    ft_vec *= ft_max / ft_mag

                f_total = fn * n_wall + ft_vec
                f[i] += f_total

                # Torque on particle:
                # r_c x F_t = (-r_i * n_wall) x ft_vec = -r_i * (n_wall x ft_vec)
                torque[i] += -r_i * n_wall.cross(ft_vec)

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.post_force_kernel(
            atom.nlocal,
            dt,
            self.wall_axis,
            self.wall_side,
            self.wall_coord,
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
        )
