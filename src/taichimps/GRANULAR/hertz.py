"""
LAMMPS pair_style gran/hertz (nonlinear contact without shear history) in Taichi.
Reference: LAMMPS src/GRANULAR/pair_gran_hertz.cpp
License: GPL v2 / taichimps MIT reimplementation
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.GRANULAR.base import GranularPair
from taichimps.neighbor import NeighborList


@ti.data_oriented
class GranHertz(GranularPair):
    """
    Hertzian nonlinear contact force without history.
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
        # LAMMPS zeroes gammat when dampflag == 0; damping is always meff-scaled.
        self.gammat = 0.0 if self.dampflag == 0 else float(gammat)
        self.xmu = float(xmu)
        self.limit_damping = 1 if limit_damping else 0

    @ti.kernel
    def compute_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
        virial: ti.template(),
        npairs: ti.template(),
        pair_i: ti.template(),
        pair_j: ti.template(),
    ):
            # One thread per candidate contact, not per particle: at this
            # system size the per-particle form leaves the GPU mostly idle
            # (1400 threads) and unbalanced (0..max_neighbors partners each).
            # See PairGranular.compute_kernel for the measurements.
        for nc in range(npairs[None]):
            i = pair_i[nc]
            j = pair_j[nc]
            xi = x[i]
            vi = v[i]
            ri = radius[i]
            mi = rmass[i]
            oi = omega[i]
            xj = x[j]
            vj = v[j]
            rj = radius[j]
            mj = rmass[j]
            oj = omega[j]

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
                vrel = vtr.norm()

                damp = meff * self.gamman * vnnr * rsqinv
                ccel = (self.kn * delta * rinv - damp) * polyhertz
                if self.limit_damping == 1 and ccel < 0.0:
                    ccel = 0.0

                fn_coulomb = self.xmu * ti.abs(ccel * r)
                fs_damp = meff * self.gammat * vrel

                ft = 0.0
                if vrel > 1e-16:
                    ft = ti.min(fn_coulomb, fs_damp) / vrel

                fs_vec = -ft * vtr
                f_total = dpos * ccel + fs_vec

                ti.atomic_add(f[i], f_total)
                ti.atomic_add(f[j], -f_total)

                tor = rinv * dpos.cross(fs_vec)
                ti.atomic_add(torque[i], -ri * tor)
                ti.atomic_add(torque[j], -rj * tor)

                # Pairwise virial, as in Pair::ev_tally_xyz()
                vir = 0.5 * ti.Vector([
                    dpos[0] * f_total[0],
                    dpos[1] * f_total[1],
                    dpos[2] * f_total[2],
                    dpos[0] * f_total[1],
                    dpos[0] * f_total[2],
                    dpos[1] * f_total[2],
                ])
                ti.atomic_add(virial[i], vir)
                ti.atomic_add(virial[j], vir)

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
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
            atom.virial,
            nlist.npairs,
            nlist.pair_i,
            nlist.pair_j,
        )
