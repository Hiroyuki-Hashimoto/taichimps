"""
LAMMPS fix wall/gran/region cylinder implementation in Taichi.
Reference: LAMMPS src/GRANULAR/fix_wall_gran_region.cpp, region_cylinder.cpp
License: GPL v2 / taichimps MIT reimplementation
"""

from typing import Any, Literal

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixWallGranRegion(Fix):
    """
    Granular wall contact with a cylinder region.
    Supports:
      - cylinder axis along x, y, or z (0, 1, or 2)
      - center coordinates (c1, c2) in the cross section
      - radius R
      - internal wall contact (side="in" or side=-1) or external wall contact (side="out" or side=1)
      - optional cutoff range along axis [axis_lo, axis_hi]
      - Hooke or Hertz normal contact
      - Normal damping
      - Tangential damping / Coulomb friction limit
      - Torque generation on particle
    """

    def __init__(
        self,
        domain: Domain,
        axis: int | str = 2,
        c1: float = 0.0,
        c2: float = 0.0,
        radius: float = 1.0,
        side: Literal["in", "out", -1, 1] = "in",
        axis_lo: float | None = None,
        axis_hi: float | None = None,
        fstyle: str = "hooke",
        kn: float = 1e5,
        gamman: float = 0.0,
        kt: float = 0.0,
        gammat: float = 0.0,
        xmu: float = 0.0,
        dampflag: int = 1,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain)
        if isinstance(axis, str):
            axis = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        self.axis = int(axis)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.radius = float(radius)

        # side: -1 for "in" (particles inside cylinder, contact with inner wall, normal points inward towards axis)
        #       +1 for "out" (particles outside cylinder, contact with outer wall, normal points outward away from axis)
        if isinstance(side, str):
            self.side = -1 if side.lower() in ("in", "inside", "-1") else 1
        else:
            self.side = -1 if side < 0 else 1

        self.axis_lo = float(axis_lo) if axis_lo is not None else -1e30
        self.axis_hi = float(axis_hi) if axis_hi is not None else 1e30
        self.has_axis_bounds = 1 if (axis_lo is not None or axis_hi is not None) else 0

        self.fstyle = fstyle.lower()
        self.is_hertz = 1 if "hertz" in self.fstyle else 0

        self.kn = float(kn)
        self.gamman = float(gamman)
        self.kt = float(kt) if kt is not None else (2.0 / 7.0 * self.kn)
        self.gammat = float(gammat) if gammat is not None else (0.5 * self.gamman)
        self.xmu = float(xmu)
        self.dampflag = int(dampflag)
        self.float_type = float_type

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        dt: ti.template(),
        axis: ti.i32,
        c1: ti.template(),
        c2: ti.template(),
        radius_cyl: ti.template(),
        side: ti.i32,
        has_axis_bounds: ti.i32,
        axis_lo: ti.template(),
        axis_hi: ti.template(),
        is_hertz: ti.i32,
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
    ):
        for i in range(nlocal):
            pos = x[i]
            r_i = radius[i]

            pos_axis = pos[axis]
            if has_axis_bounds == 1 and (pos_axis < axis_lo or pos_axis > axis_hi):
                continue

            # Determine coordinates in the cross-section plane
            # axis=0 (x-cylinder): cross section (y, z)
            # axis=1 (y-cylinder): cross section (x, z)
            # axis=2 (z-cylinder): cross section (x, y)
            d1 = 0.0
            d2 = 0.0
            if axis == 0:
                d1 = pos[1] - c1
                d2 = pos[2] - c2
            elif axis == 1:
                d1 = pos[0] - c1
                d2 = pos[2] - c2
            else:
                d1 = pos[0] - c1
                d2 = pos[1] - c2

            r_perp = ti.sqrt(d1 * d1 + d2 * d2)
            if r_perp > 1e-14:
                inv_r_perp = 1.0 / r_perp
                u1 = d1 * inv_r_perp
                u2 = d2 * inv_r_perp

                # Unit normal pointing AWAY from the wall (into the particle contact region):
                # For side == -1 ("in"): particles inside, wall is at r_perp = radius_cyl.
                # Normal pointing inward (towards axis) = -[u1, u2] in cross section.
                # For side == +1 ("out"): particles outside, wall is at r_perp = radius_cyl.
                # Normal pointing outward (away from axis) = +[u1, u2] in cross section.
                n_wall = ti.Vector([0.0, 0.0, 0.0])
                if side == -1:
                    # Inside cylinder: normal is -radial
                    if axis == 0:
                        n_wall = ti.Vector([0.0, -u1, -u2])
                    elif axis == 1:
                        n_wall = ti.Vector([-u1, 0.0, -u2])
                    else:
                        n_wall = ti.Vector([-u1, -u2, 0.0])
                else:
                    # Outside cylinder: normal is +radial
                    if axis == 0:
                        n_wall = ti.Vector([0.0, u1, u2])
                    elif axis == 1:
                        n_wall = ti.Vector([u1, 0.0, u2])
                    else:
                        n_wall = ti.Vector([u1, u2, 0.0])

                # Overlap delta:
                # Inside cylinder: distance to wall surface = radius_cyl - r_perp.
                # Overlap delta = r_i - (radius_cyl - r_perp) = r_i - radius_cyl + r_perp
                # Outside cylinder: distance to wall surface = r_perp - radius_cyl.
                # Overlap delta = r_i - (r_perp - radius_cyl) = r_i - r_perp + radius_cyl
                delta = 0.0
                if side == -1:
                    delta = r_i - (radius_cyl - r_perp)
                else:
                    delta = r_i - (r_perp - radius_cyl)

                if delta > 0.0:
                    mi = rmass[i]
                    vi = v[i]
                    oi = omega[i]

                    # Relative velocity at contact point
                    # Contact point on particle relative to center is -r_i * n_wall
                    # Surface velocity of particle: vi + oi x (-r_i * n_wall) = vi - r_i * (oi x n_wall)
                    # Wall is stationary: v_wall = 0
                    vr = vi - r_i * oi.cross(n_wall)
                    vn = vr.dot(n_wall)
                    vrn = vn * n_wall
                    vrt = vr - vrn

                    # Normal force computation (Hooke vs Hertz)
                    fn = 0.0
                    fn_elastic = 0.0
                    if is_hertz == 1:
                        polyhertz = ti.sqrt(r_i * delta)
                        fn_elastic = self.kn * delta * polyhertz
                        fn_damp = self.gamman * vn * polyhertz
                        if self.dampflag == 1:
                            fn_damp *= mi
                        fn = fn_elastic - fn_damp
                    else:
                        fn_elastic = self.kn * delta
                        fn_damp = self.gamman * vn
                        if self.dampflag == 1:
                            fn_damp *= mi
                        fn = fn_elastic - fn_damp

                    fn = max(fn, 0.0)

                    # Tangential force computation
                    ft_damp = self.gammat * vrt
                    if is_hertz == 1:
                        polyhertz = ti.sqrt(r_i * delta)
                        ft_damp *= polyhertz
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
                    # Torque = r_c x F = (-r_i * n_wall) x ft_vec = -r_i * (n_wall x ft_vec)
                    torque[i] += -r_i * n_wall.cross(ft_vec)

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return
        self.post_force_kernel(
            atom.nlocal,
            dt,
            self.axis,
            self.c1,
            self.c2,
            self.radius,
            self.side,
            self.has_axis_bounds,
            self.axis_lo,
            self.axis_hi,
            self.is_hertz,
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
        )
