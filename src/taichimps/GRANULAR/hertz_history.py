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
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
        tag: ti.template(),
        virial: ti.template(),
        num_neighbors: ti.template(),
        neighbors: ti.template(),
        shear_hist: ti.template(),
        partner_hist: ti.template(),
    ):
        for i in range(nlocal):
            xi = x[i]
            vi = v[i]
            ri = radius[i]
            mi = rmass[i]
            oi = omega[i]

            n_neigh = num_neighbors[i]
            for k in range(n_neigh):
                j = neighbors[i, k]
                if j < 0:
                    continue

                xj = x[j]
                vj = v[j]
                rj = radius[j]
                mj = rmass[j]
                oj = omega[j]

                dpos = self.domain.minimum_image(xi - xj)
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
                    vnnr = (vi - vj).dot(dpos)
                    vn_vec = dpos * (vnnr * rsqinv)
                    vt_vec = (vi - vj) - vn_vec
                    wr = (ri * oi + rj * oj) * rinv
                    # LAMMPS: vtr1 = vt1 - (delz*wr2 - dely*wr3), i.e. vt - W x n.
                    # dpos.cross(wr) = n x W = -(W x n), hence the plus sign.
                    vtr = vt_vec + dpos.cross(wr)

                    damp = meff * self.gamman * vnnr * rsqinv
                    ccel = (self.kn * delta * rinv - damp) * polyhertz
                    if self.limit_damping == 1 and ccel < 0.0:
                        ccel = 0.0

                    jtag = tag[j]
                    shear = shear_hist[i, k]
                    if partner_hist[i, k] != jtag:
                        shear = ti.Vector([0.0, 0.0, 0.0])
                        partner_hist[i, k] = jtag

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

                    shear_hist[i, k] = shear

                    f_total = dpos * ccel + fs_vec

                    ti.atomic_add(f[i], f_total)
                    ti.atomic_add(f[j], -f_total)

                    # Legacy gran/hertz/history uses the full radii as the
                    # moment arm; the (radi - delta/2) form belongs to the
                    # newer pair granular (GranularModel::calculate_forces).
                    tor = rinv * dpos.cross(fs_vec)
                    ti.atomic_add(torque[i], -ri * tor)
                    ti.atomic_add(torque[j], -rj * tor)

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
                else:
                    partner_hist[i, k] = -1
                    shear_hist[i, k] = ti.Vector([0.0, 0.0, 0.0])

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
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
            atom.tag,
            atom.virial,
            nlist.num_neighbors,
            nlist.neighbors,
            history.shear,
            history.partner,
        )
