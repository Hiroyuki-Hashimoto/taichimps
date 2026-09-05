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
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        omega: ti.template(),
        torque: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
        num_neighbors: ti.template(),
        neighbors: ti.template(),
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
                    vtr = vt_vec - dpos.cross(wr)
                    vrel = vtr.norm()

                    damp = meff * self.gamman * vnnr * rsqinv
                    if self.dampflag == 0:
                        damp = self.gamman * vnnr * rsqinv

                    ccel = (self.kn * delta * rinv - damp) * polyhertz
                    ccel = max(ccel, 0.0)

                    fn_coulomb = self.xmu * ti.abs(ccel * r)
                    fs_damp = meff * self.gammat * vrel
                    if self.dampflag == 0:
                        fs_damp = self.gammat * vrel

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
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
            nlist.num_neighbors,
            nlist.neighbors,
        )
