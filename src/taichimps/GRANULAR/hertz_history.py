"""
LAMMPS pair_style gran/hertz/history implementation in Taichi.
Reference: LAMMPS src/GRANULAR/pair_gran_hertz_history.cpp
License: GPL v2 / taichimps MIT reimplementation

Hertzian contact theory for spheres:
Effective radius: Reff = ri * rj / (ri + rj)
Effective mass: meff = mi * mj / (mi + mj)
Normal stiffness: Kn_eff = Kn * sqrt(Reff * delta)
Normal damping: Gamman_eff = Gamman * sqrt(Reff * delta) * meff
Tangential stiffness: Kt_eff = Kt * sqrt(Reff * delta)
Tangential damping: Gammat_eff = Gammat * sqrt(Reff * delta) * meff

Note on `dampflag`: as in LAMMPS, damping is always proportional to meff and
`dampflag = 0` simply zeroes gammat (PairGranHookeHistory::settings), so it is
applied in __init__ rather than branched on inside the kernel.
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.GRANULAR.base import GranularPair
from taichimps.neighbor import NeighborList


@ti.data_oriented
class GranHertzHistory(GranularPair):
    """
    Hertzian nonlinear normal force + nonlinear tangential history with Coulomb friction.
    """

    def __init__(
        self,
        domain: Domain,
        kn: float,
        gamman: float,
        kt: float,
        gammat: float,
        xmu: float,
        dampflag: int = 1,
        limit_damping: bool = False,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)
        self.kn = float(kn)
        self.gamman = float(gamman)
        self.kt = float(kt)
        self.dampflag = int(dampflag)
        self.gammat = 0.0 if self.dampflag == 0 else float(gammat)
        self.xmu = float(xmu)
        self.limit_damping = 1 if limit_damping else 0

    @ti.kernel
    def compute_kernel(
        self,
        nlocal: ti.i32,
        dt: ti.template(),
        shearupdate: ti.i32,
        atom: ti.template(),
        npairs: ti.template(),
        pair_i: ti.template(),
        pair_j: ti.template(),
        shear_hist: ti.template(),
        partner_hist: ti.template(),
    ):
            # One thread per candidate contact, not per particle: at this
            # system size the per-particle form leaves the GPU mostly idle
            # (1400 threads) and unbalanced (0..max_neighbors partners each).
            # See PairGranular.compute_kernel for the measurements.
        for nc in range(npairs[None]):
            i = pair_i[nc]
            j = pair_j[nc]
            xi = atom.x[i]
            vi = atom.v[i]
            ri = atom.radius[i]
            mi = atom.rmass[i]
            oi = atom.omega[i]
            xj = atom.x[j]
            vj = atom.v[j]
            rj = atom.radius[j]
            mj = atom.rmass[j]
            oj = atom.omega[j]

            dpos, dvj = self.domain.minimum_image_and_vshift(xi - xj)
            rsq = dpos.dot(dpos)
            radsum = ri + rj

            if rsq < radsum * radsum and rsq > 1e-28:
                r = ti.sqrt(rsq)
                delta = radsum - r
                reff = (ri * rj) / (ri + rj)
                meff = (mi * mj) / (mi + mj)
                polyhertz = ti.sqrt(reff * delta)

                rsqinv = 1.0 / rsq
                rinv = 1.0 / r
                # Under `remap v` the periodic image of j moves with the
                # deforming lattice, so its velocity is offset.
                vrel_t = vi - (vj + dvj)
                vnnr = vrel_t.dot(dpos)
                vn_vec = dpos * (vnnr * rsqinv)
                vt_vec = vrel_t - vn_vec
                wr = (ri * oi + rj * oj) * rinv
                # LAMMPS: vtr1 = vt1 - (delz*wr2 - dely*wr3), i.e. vt - W x n.
                # dpos.cross(wr) = n x W = -(W x n), hence the plus sign.
                vtr = vt_vec + dpos.cross(wr)

                damp = meff * self.gamman * vnnr * rsqinv
                ccel = (self.kn * delta * rinv - damp) * polyhertz
                if self.limit_damping == 1 and ccel < 0.0:
                    ccel = 0.0

                jtag = atom.tag[j]
                shear = shear_hist[nc]
                if partner_hist[nc] != jtag:
                    shear = ti.Vector([0.0, 0.0, 0.0])
                    partner_hist[nc] = jtag

                gammat_eff = meff * self.gammat

                if shearupdate != 0:
                    shear = shear + vtr * dt
                shrmag = shear.norm()
                if shearupdate != 0:
                    rsht = shear.dot(dpos) * rsqinv
                    shear = shear - rsht * dpos

                fs_vec = -polyhertz * (self.kt * shear + gammat_eff * vtr)
                fs_mag = fs_vec.norm()

                fn_coulomb = self.xmu * ti.abs(ccel * r)
                if fs_mag > fn_coulomb:
                    if shrmag != 0.0:
                        ratio = fn_coulomb / fs_mag
                        damp_corr = gammat_eff * vtr / self.kt
                        shear = ratio * (shear + damp_corr) - damp_corr
                        fs_vec = ratio * fs_vec
                    else:
                        fs_vec = ti.Vector([0.0, 0.0, 0.0])

                shear_hist[nc] = shear

                f_total = dpos * ccel + fs_vec

                ti.atomic_add(atom.f[i], f_total)
                ti.atomic_add(atom.f[j], -f_total)

                # Legacy gran/hertz/history uses the full radii as the
                # moment arm; the (radi - delta/2) form belongs to the
                # newer pair granular (GranularModel::calculate_forces).
                tor = rinv * dpos.cross(fs_vec)
                ti.atomic_add(atom.torque[i], -ri * tor)
                ti.atomic_add(atom.torque[j], -rj * tor)

                vir = 0.5 * ti.Vector([
                    dpos[0] * f_total[0],
                    dpos[1] * f_total[1],
                    dpos[2] * f_total[2],
                    dpos[0] * f_total[1],
                    dpos[0] * f_total[2],
                    dpos[1] * f_total[2],
                ])
                ti.atomic_add(atom.virial[i], vir)
                ti.atomic_add(atom.virial[j], vir)
            else:
                partner_hist[nc] = -1
                shear_hist[nc] = ti.Vector([0.0, 0.0, 0.0])

    def compute(
        self,
        atom: AtomSystem,
        nlist: NeighborList,
        history: ContactHistory,
        dt: float,
        shearupdate: bool = True,
    ) -> None:
        if atom.nlocal == 0:
            return
        self.compute_kernel(
            atom.nlocal,
            dt,
            1 if shearupdate else 0,
            atom,
            nlist.npairs,
            nlist.pair_i,
            nlist.pair_j,
            history.shear,
            history.partner,
        )
