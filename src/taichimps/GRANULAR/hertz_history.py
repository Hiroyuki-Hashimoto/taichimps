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
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)
        self.kn = float(kn)
        self.gamman = float(gamman)
        self.kt = float(kt)
        self.gammat = float(gammat)
        self.xmu = float(xmu)
        self.dampflag = dampflag

    @ti.kernel
    def compute_kernel(
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
                    vtr = vt_vec - dpos.cross(wr) # dpos x wr matches LAMMPS (delz*wr2-dely*wr3)

                    damp = meff * self.gamman * vnnr * rsqinv
                    if self.dampflag == 0:
                        damp = self.gamman * vnnr * rsqinv

                    ccel = (self.kn * delta * rinv - damp) * polyhertz
                    ccel = max(ccel, 0.0)

                    partner = partner_hist[i, k]
                    shear = shear_hist[i, k]
                    if partner != j:
                        shear = ti.Vector([0.0, 0.0, 0.0])
                        partner_hist[i, k] = j

                    shear = shear + vtr * dt

                    # Rotate shear displacements: shear -= (shear . dpos) * rsqinv * dpos
                    rsht = shear.dot(dpos) * rsqinv
                    shear = shear - rsht * dpos
                    shrmag = shear.norm()

                    gammat_eff = meff * self.gammat if self.dampflag == 1 else self.gammat
                    fs_vec = -polyhertz * (self.kt * shear + gammat_eff * vtr)
                    fs_mag = fs_vec.norm()

                    fn_coulomb = self.xmu * ti.abs(ccel * r)
                    if fs_mag > fn_coulomb and shrmag > 1e-16:
                        ratio = fn_coulomb / fs_mag
                        damp_corr = gammat_eff * vtr / self.kt
                        shear = ratio * (shear + damp_corr) - damp_corr
                        fs_vec = -polyhertz * (self.kt * shear + gammat_eff * vtr)

                    shear_hist[i, k] = shear

                    f_total = dpos * ccel + fs_vec

                    ti.atomic_add(f[i], f_total)
                    ti.atomic_add(f[j], -f_total)

                    tor = rinv * dpos.cross(fs_vec)
                    radstep_i = ri - 0.5 * delta
                    radstep_j = rj - 0.5 * delta
                    ti.atomic_add(torque[i], -radstep_i * tor)
                    ti.atomic_add(torque[j], -radstep_j * tor)
                else:
                    partner_hist[i, k] = -1
                    shear_hist[i, k] = ti.Vector([0.0, 0.0, 0.0])

    def compute(
        self,
        atom: AtomSystem,
        nlist: NeighborList,
        history: ContactHistory,
        dt: float,
    ) -> None:
        if atom.nlocal == 0:
            return
        self.compute_kernel(
            atom.nlocal,
            dt,
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
            nlist.num_neighbors,
            nlist.neighbors,
            history.shear,
            history.partner,
        )
