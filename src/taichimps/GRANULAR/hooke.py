"""
LAMMPS pair_style gran/hooke (contact without shear history) implementation in Taichi.
Reference: LAMMPS src/GRANULAR/pair_gran_hooke.cpp
License: GPL v2 / taichimps MIT reimplementation

Normal force:
  Fn = Kn * delta - Gamman * meff * vn
Tangential force:
  Ft = - Gammat * meff * vrt  (pure viscous tangential damping, bounded by mu * Fn)
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.GRANULAR.base import GranularPair
from taichimps.neighbor import NeighborList


@ti.data_oriented
class GranHooke(GranularPair):
    """
    Hookean normal force + tangential viscous damping without history.
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
        # LAMMPS: "if (dampflag == 0) gammat = 0.0". Damping itself is always
        # proportional to meff; dampflag only excludes the tangential term.
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

                    meff = (mi * mj) / (mi + mj)

                    # LAMMPS C++ pair_gran_hooke.cpp:
                    #   rinv = 1.0/r; rsqinv = 1.0/rsq;
                    #   vr1 = v[i][0] - v[j][0]; ...
                    #   vnnr = vr1*delx + vr2*dely + vr3*delz;
                    #   vn1 = delx*vnnr * rsqinv; ...
                    #   vt1 = vr1 - vn1; ...
                    #   wr1 = (radi*omega[i][0] + radj*omega[j][0]) * rinv; ...
                    #   vtr1 = vt1 - (delz*wr2-dely*wr3); ...
                    #   damp = meff*gamman*vnnr*rsqinv;
                    #   ccel = kn*(radsum-r)*rinv - damp;
                    #   fn = xmu * fabs(ccel*r);
                    #   fs = meff*gammat*vrel;
                    #   ft = MIN(fn,fs) / vrel;
                    #   fs1 = -ft*vtr1; ...
                    #   fx = delx*ccel + fs1; ...
                    #   tor1 = rinv * (dely*fs3 - delz*fs2);
                    #   torque[i] -= radi * tor1; torque[j] -= radj * tor1;
                    rsqinv = 1.0 / rsq
                    rinv = 1.0 / r
                    vnnr = (vi - vj).dot(dpos)
                    vn_vec = dpos * (vnnr * rsqinv)
                    vt_vec = (vi - vj) - vn_vec
                    wr = (ri * oi + rj * oj) * rinv
                    # LAMMPS: vtr1 = vt1 - (delz*wr2 - dely*wr3), i.e. vt - W x n.
                    # dpos.cross(wr) = n x W = -(W x n), hence the plus sign.
                    vtr = vt_vec + dpos.cross(wr)
                    vrel = vtr.norm()

                    damp = meff * self.gamman * vnnr * rsqinv
                    ccel = self.kn * delta * rinv - damp
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

                    # LAMMPS torque calculation:
                    # tor1 = rinv * (dely*fs3 - delz*fs2); ...
                    # tor = rinv * (dpos x fs_vec)
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
            nlist.num_neighbors,
            nlist.neighbors,
        )
