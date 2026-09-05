"""
LAMMPS pair_style gran/hooke/history implementation in Taichi.
Reference: LAMMPS src/GRANULAR/pair_gran_hooke_history.cpp
License: GPL v2 / taichimps MIT reimplementation

Formulation:
Normal force:
  Fn = - (Kn * delta - damp_n * meff * vn) * n_ij
Tangential force (incremental history):
  vtr = vt - (r_i * omega_i + r_j * omega_j) x n_ij
  vrel = (v_i - v_j) + (r_i * omega_i + r_j * omega_j) x n_ij
  delta_s += vrel_tangential * dt
  Ft_trial = - (Kt * delta_s + damp_t * meff * vt)
Coulomb criterion:
  if |Ft_trial| > mu * |Fn_elastic|:
      Ft = Ft_trial * (mu * |Fn_elastic| / |Ft_trial|)
      delta_s = -Ft / Kt  (rescale shear displacement)
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.GRANULAR.base import GranularPair
from taichimps.neighbor import NeighborList


@ti.data_oriented
class GranHookeHistory(GranularPair):
    """
    Hookean normal force + tangential contact history with Coulomb friction.
    Parameters:
      kn: Normal spring stiffness
      gamman: Normal damping constant (or damping factor if dampflag=0)
      kt: Tangential spring stiffness
      gammat: Tangential damping constant
      xmu: Friction coefficient
      dampflag: 1 if damping is proportional to meff (standard), 0 if absolute
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

                # Distance and normal vector
                dpos = self.domain.minimum_image(xi - xj)
                rsq = dpos.dot(dpos)
                radsum = ri + rj

                if rsq < radsum * radsum and rsq > 1e-28:
                    r = ti.sqrt(rsq)
                    delta = radsum - r  # Overlap (> 0)

                    # Effective mass meff = mi * mj / (mi + mj)
                    meff = (mi * mj) / (mi + mj)

                    # Relative contact velocity:
                    # vr = (vi - vj) + (ri * oi + rj * oj) x n
                    # Notice rotation sign: at contact point between i and j:
                    # Contact point on i: xi - ri * n
                    # Contact point on j: xj + rj * n
                    # Relative surface velocity v_contact_rel = (vi - ri * (oi x n)) - (vj + rj * (oj x n))
                    # = (vi - vj) - (ri * oi + rj * oj) x n
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

                    ccel = self.kn * delta * rinv - damp
                    ccel = max(ccel, 0.0)

                    # Tangential force (incremental shear history)
                    partner = partner_hist[i, k]
                    shear = shear_hist[i, k]
                    if partner != j:
                        shear = ti.Vector([0.0, 0.0, 0.0])
                        partner_hist[i, k] = j

                    shear = shear + vtr * dt

                    # Rotate shear displacement vector: shear -= (shear . dpos) * rsqinv * dpos
                    rsht = shear.dot(dpos) * rsqinv
                    shear = shear - rsht * dpos
                    shrmag = shear.norm()

                    # Tangential force components:
                    gammat_eff = meff * self.gammat if self.dampflag == 1 else self.gammat
                    fs_vec = -(self.kt * shear + gammat_eff * vtr)
                    fs_mag = fs_vec.norm()

                    fn_coulomb = self.xmu * ti.abs(ccel * r)
                    if fs_mag > fn_coulomb and shrmag > 1e-16:
                        ratio = fn_coulomb / fs_mag
                        damp_corr = gammat_eff * vtr / self.kt
                        shear = ratio * (shear + damp_corr) - damp_corr
                        fs_vec = -(self.kt * shear + gammat_eff * vtr)

                    shear_hist[i, k] = shear

                    f_total = dpos * ccel + fs_vec

                    ti.atomic_add(f[i], f_total)
                    ti.atomic_add(f[j], -f_total)

                    tor = rinv * dpos.cross(fs_vec)
                    ti.atomic_add(torque[i], -ri * tor)
                    ti.atomic_add(torque[j], -rj * tor)
                else:
                    # Not in contact: clear history
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
