"""
LAMMPS fix wall/gran/region cylinder implementation in Taichi.
Reference: LAMMPS src/GRANULAR/fix_wall_gran_region.cpp, region_cylinder.cpp
License: GPL v2 / taichimps MIT reimplementation

The wall is a contact partner of zero radius and infinite mass, exactly as for
the plane wall: Reff = radi, meff = rmass[i], contact radius a = sqrt(delta*radi)
and the moment arm is the full particle radius.

This used to have no shear history -- a viscous tangential dashpot with a
Coulomb cap, i.e. the history-free `hooke`/`hertz` styles -- so a specimen
resting against a cylindrical wall had no frictional memory at the boundary.
`history=True` (the default, and what the `*/history` style names select) keeps
a shear displacement per particle.

Limitation: LAMMPS lets one particle touch several surfaces of a region at once
and keeps a history entry per surface (history_many[i][iwall]). Only the
cylinder's curved side is treated here, so there is one history slot per
particle. `axis_lo`/`axis_hi` bound which particles are considered; they are not
end-cap walls.
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
        kt: float | None = None,
        gammat: float | None = None,
        xmu: float = 0.0,
        dampflag: int = 1,
        history: bool | None = None,
        limit_damping: bool = False,
        max_atoms: int = 100000,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)
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
        self.dampflag = int(dampflag)
        # LAMMPS: "if (dampflag == 0) gammat = 0.0". Damping is always scaled by
        # the effective mass; dampflag only excludes the tangential term.
        gammat_val = float(gammat) if gammat is not None else (0.5 * self.gamman)
        self.gammat = 0.0 if self.dampflag == 0 else gammat_val
        self.xmu = float(xmu)
        self.limit_damping = 1 if limit_damping else 0
        # The style name decides unless the caller overrides it explicitly.
        self.use_history = (
            1 if (self.fstyle.endswith("history") if history is None else history) else 0
        )
        self.float_type = float_type

        # One history slot per particle: the cylinder side is a single surface.
        self.shear = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.touch = ti.field(dtype=ti.i32, shape=max_atoms)

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
        history_update: ti.i32,
        atom: ti.template(),
    ):
        for i in range(nlocal):
            pos = atom.x[i]
            r_i = atom.radius[i]

            pos_axis = pos[axis]
            if has_axis_bounds == 1 and (pos_axis < axis_lo or pos_axis > axis_hi):
                if ti.static(self.use_history == 1):
                    self.touch[i] = 0
                    self.shear[i] = ti.Vector([0.0, 0.0, 0.0])
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
                    mi = atom.rmass[i]
                    vi = atom.v[i]
                    oi = atom.omega[i]

                    # Relative velocity at the contact point. The wall is
                    # stationary and has zero radius, so W = radi * omega_i;
                    # (omega x n) is perpendicular to n, so vn is unaffected by
                    # subtracting it here and vrt comes out as vt - W x n.
                    vr = vi - r_i * oi.cross(n_wall)
                    vn = vr.dot(n_wall)
                    vrt = vr - vn * n_wall
                    vrel = vrt.norm()

                    # Contact radius for the Hertz variants: a = sqrt(delta*Reff)
                    # with Reff = radi, since the wall has zero curvature here.
                    poly = 1.0
                    if is_hertz == 1:
                        poly = ti.sqrt(r_i * delta)

                    fn_elastic = self.kn * delta * poly
                    # hooke -> mass_velocity damping, hertz -> viscoelastic,
                    # which carries the extra contact-radius factor.
                    damp_prefactor = mi * self.gamman * poly
                    fntot = fn_elastic - damp_prefactor * vn
                    if ti.static(self.limit_damping == 1):
                        fntot = ti.max(fntot, 0.0)
                    # LAMMPS bounds friction by the TOTAL normal force.
                    fscrit = self.xmu * ti.abs(fntot)

                    damp_t = mi * self.gammat * poly
                    ft_vec = ti.Vector([0.0, 0.0, 0.0])

                    if ti.static(self.use_history == 1):
                        hist = self.shear[i]
                        if self.touch[i] == 0:
                            hist = ti.Vector([0.0, 0.0, 0.0])
                            self.touch[i] = 1

                        # linear_history_classic: accumulate, take |history|,
                        # then project back into the tangent plane.
                        if history_update != 0:
                            hist = hist + vrt * dt
                        shrmag = hist.norm()
                        if history_update != 0:
                            hist = hist - hist.dot(n_wall) * n_wall

                        fdamp = -damp_t * vrt
                        ft_vec = -self.kt * poly * hist + fdamp

                        magfs = ft_vec.norm()
                        if magfs > fscrit:
                            if shrmag != 0.0:
                                ft_vec = ft_vec * (fscrit / magfs)
                                hist = -(ft_vec - fdamp) / (self.kt * poly)
                            else:
                                ft_vec = ti.Vector([0.0, 0.0, 0.0])
                        self.shear[i] = hist
                    else:
                        ft = 0.0
                        if vrel != 0.0:
                            ft = ti.min(fscrit, damp_t * vrel) / vrel
                        ft_vec = -ft * vrt

                    atom.f[i] += fntot * n_wall + ft_vec
                    # For a wall the moment arm is the full radius.
                    atom.torque[i] += -r_i * n_wall.cross(ft_vec)
                else:
                    if ti.static(self.use_history == 1):
                        self.touch[i] = 0
                        self.shear[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                # On the cylinder axis: no well-defined normal, no contact.
                if ti.static(self.use_history == 1):
                    self.touch[i] = 0
                    self.shear[i] = ti.Vector([0.0, 0.0, 0.0])

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
            1 if self.history_update else 0,
            atom,
        )
