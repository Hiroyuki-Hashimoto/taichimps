"""
Frictional contact with a flat, axis-aligned wall.

Reference: LAMMPS src/GRANULAR/fix_wall_gran.cpp
License: GPL v2 / taichimps MIT reimplementation

LAMMPS treats the wall as a contact partner with zero radius and infinite mass,
which fixes the geometry of the interaction:

    radsum = radi,  Reff = radi,  meff = rmass[i],  vj = vwall,  omegaj = 0

and the moment arm is the full particle radius (GranularModel scales torquesi by
radi for contact_type == WALL, not by radi - delta/2 as it does for a pair).

`fix wall/gran hooke/history Kn Kt gamma_n gamma_t xmu dampflag` expands to a
hooke normal model, mass_velocity damping and the linear_history_classic
tangential model (GranularModel::define_classic_model), which is the same
algebra as pair gran/hooke/history.  That is what is implemented here.

This used to have no shear history at all -- just a viscous tangential dashpot
with a Coulomb cap, i.e. the history-free `hooke` style -- so a specimen resting
against walls had no frictional memory at the boundary.  `history=True` (the
default) now keeps a shear displacement per particle per wall.
"""

import math
from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class FixWallGran(Fix):
    """
    Granular contact with one axis-aligned plane wall.

    Parameters
        wall_axis   0, 1 or 2
        wall_side   -1 for a wall below the particles (normal along +axis),
                    +1 for a wall above them (normal along -axis)
        wall_coord  position of the wall plane
        kn, kt      normal and tangential stiffness
        gamman, gammat  damping coefficients (always scaled by the particle mass,
                    as in LAMMPS; dampflag = 0 zeroes gammat rather than
                    removing the mass scaling)
        xmu         friction coefficient
        history     keep tangential shear history (the `hooke/history` style);
                    False reproduces the history-free `hooke` style
        shear_axis / shear_vel  in-plane wall velocity (`shear` keyword)
        wiggle_amplitude / wiggle_period  oscillation of the wall along its
                    normal (`wiggle` keyword)
    """

    def __init__(
        self,
        domain: Domain,
        wall_axis: int,
        wall_side: int,
        wall_coord: float,
        kn: float,
        gamman: float,
        kt: float,
        gammat: float,
        xmu: float,
        dampflag: int = 1,
        history: bool = True,
        limit_damping: bool = False,
        shear_axis: int = -1,
        shear_vel: float = 0.0,
        wiggle_amplitude: float = 0.0,
        wiggle_period: float = 0.0,
        max_atoms: int = 100000,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)
        if wall_axis not in (0, 1, 2):
            raise ValueError("wall_axis must be 0, 1 or 2")
        if wall_side not in (-1, 1):
            raise ValueError("wall_side must be -1 (low) or +1 (high)")
        if shear_axis == wall_axis:
            raise ValueError("shear must act in the plane of the wall, not along its normal")
        if wiggle_amplitude != 0.0 and shear_vel != 0.0:
            raise ValueError("cannot wiggle and shear the same wall, as in LAMMPS")
        if wiggle_amplitude != 0.0 and wiggle_period <= 0.0:
            raise ValueError("wiggle needs a positive period")

        self.wall_axis = int(wall_axis)
        self.wall_side = int(wall_side)
        self.wall_coord = float(wall_coord)
        self.kn = float(kn)
        self.gamman = float(gamman)
        self.kt = float(kt)
        self.dampflag = int(dampflag)
        # LAMMPS: "if (dampflag == 0) gammat = 0.0"
        self.gammat = 0.0 if self.dampflag == 0 else float(gammat)
        self.xmu = float(xmu)
        self.limit_damping = 1 if limit_damping else 0
        self.use_history = 1 if history else 0

        self.shear_axis = int(shear_axis)
        self.shear_vel = float(shear_vel)
        self.wiggle_amplitude = float(wiggle_amplitude)
        self.wiggle_period = float(wiggle_period)

        # Per-particle shear history for this wall. The wall is a single
        # partner, so one slot per particle is enough (LAMMPS uses a one-sided
        # FixNeighHistory for the same reason).
        self.shear = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.touch = ti.field(dtype=ti.i32, shape=max_atoms)

        self.elapsed = 0.0

    @ti.kernel
    def reset_history(self, nlocal: ti.i32):
        for i in range(nlocal):
            self.shear[i] = ti.Vector([0.0, 0.0, 0.0])
            self.touch[i] = 0

    @ti.kernel
    def post_force_kernel(
        self,
        nlocal: ti.i32,
        dt: ti.template(),
        wall_coord: ti.f64,
        vwall_n: ti.f64,
        vwall_s: ti.f64,
        history_update: ti.i32,
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
    ):
        axis = ti.static(self.wall_axis)
        side = ti.static(self.wall_side)
        shear_axis = ti.static(self.shear_axis)

        # Normal pointing from the wall into the domain.
        n = ti.Vector([0.0, 0.0, 0.0])
        n[axis] = -1.0 if side == 1 else 1.0

        vwall = ti.Vector([0.0, 0.0, 0.0])
        vwall[axis] = vwall_n
        if ti.static(shear_axis >= 0):
            vwall[shear_axis] = vwall_s

        for i in range(nlocal):
            ri = radius[i]
            # Signed distance from the wall plane along the inward normal.
            dist = (x[i][axis] - wall_coord) if side == -1 else (wall_coord - x[i][axis])
            delta = ri - dist

            if delta <= 0.0:
                if ti.static(self.use_history == 1):
                    self.touch[i] = 0
                    self.shear[i] = ti.Vector([0.0, 0.0, 0.0])
                continue

            # The wall has infinite mass, so the effective mass is the particle's.
            meff = rmass[i]

            vr = v[i] - vwall
            vnnr = vr.dot(n)
            vt = vr - vnnr * n
            # radj = 0, so W = radi * omega_i
            wr = ri * omega[i]
            vtr = vt - wr.cross(n)
            vrel = vtr.norm()

            # normal hooke + mass_velocity damping
            damp_prefactor = meff * self.gamman
            fntot = self.kn * delta - damp_prefactor * vnnr
            if ti.static(self.limit_damping == 1):
                fntot = ti.max(fntot, 0.0)
            fscrit = self.xmu * ti.abs(fntot)

            damp_t = meff * self.gammat
            fs = ti.Vector([0.0, 0.0, 0.0])

            if ti.static(self.use_history == 1):
                hist = self.shear[i]
                if self.touch[i] == 0:
                    hist = ti.Vector([0.0, 0.0, 0.0])
                    self.touch[i] = 1

                # linear_history_classic: accumulate, take |history|, then
                # project back into the tangent plane.
                if history_update != 0:
                    hist = hist + vtr * dt
                shrmag = hist.norm()
                if history_update != 0:
                    hist = hist - hist.dot(n) * n

                fdamp = -damp_t * vtr
                fs = -self.kt * hist + fdamp

                magfs = fs.norm()
                if magfs > fscrit:
                    if shrmag != 0.0:
                        fs = fs * (fscrit / magfs)
                        hist = -(fs - fdamp) / self.kt
                    else:
                        fs = ti.Vector([0.0, 0.0, 0.0])
                self.shear[i] = hist
            else:
                # linear_nohistory: viscous only, capped at the Coulomb limit
                ft = 0.0
                if vrel != 0.0:
                    ft = ti.min(fscrit, damp_t * vrel) / vrel
                fs = -ft * vtr

            f[i] += fntot * n + fs
            # For a wall the moment arm is the full radius.
            torque[i] += -ri * n.cross(fs)

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0:
            return

        # Wall position and velocity for this step, as in FixWallGran::post_force
        coord = self.wall_coord
        vwall_n = 0.0
        if self.wiggle_amplitude != 0.0:
            omega_w = 2.0 * math.pi / self.wiggle_period
            arg = omega_w * self.elapsed
            coord += self.wiggle_amplitude * (1.0 - math.cos(arg))
            vwall_n = self.wiggle_amplitude * omega_w * math.sin(arg)

        self.post_force_kernel(
            atom.nlocal,
            dt,
            coord,
            vwall_n,
            self.shear_vel,
            1 if self.history_update else 0,
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
        )
        self.elapsed += dt

    def setup(self, nsteps_total: int = 0) -> None:
        """Restart the clock the wall motion is referenced to."""
        self.elapsed = 0.0
