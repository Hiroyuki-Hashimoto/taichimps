"""
LAMMPS pair_style gran/hooke/history implementation in Taichi.
Reference: LAMMPS src/GRANULAR/pair_gran_hooke_history.cpp
License: GPL v2 / taichimps MIT reimplementation

Formulation (follows PairGranHookeHistory::compute line for line):
  damp = meff * gamma_n * vnnr / r^2
  ccel = Kn * delta / r - damp            (radial force magnitude / r)
  vtr  = vt - W x n,  W = r_i * omega_i + r_j * omega_j
  shear += vtr * dt,  then project out the component along n
  Ft   = -(Kt * shear + meff * gamma_t * vtr)
Coulomb criterion:
  Fn_crit = xmu * |ccel * r|
  if |Ft| > Fn_crit: rescale both Ft and the stored shear by Fn_crit / |Ft|

Note on `dampflag`: in LAMMPS damping is *always* proportional to meff.
`dampflag = 0` does not switch that off, it zeroes the tangential damping
coefficient outright (see PairGranHookeHistory::settings), which is why it is
applied to `gammat` in __init__ rather than branched on in the kernel.
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
      dampflag: 0 excludes the tangential damping force (gammat is zeroed), 1 includes it
      limit_damping: clamp the normal force at zero so damping cannot make the
        contact attractive. Off by default, matching the LAMMPS keyword.
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
        # LAMMPS: "if (dampflag == 0) gammat = 0.0"
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

            # Distance and normal vector
            dpos, dvj = self.domain.minimum_image_and_vshift(xi - xj)
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
                # Under `remap v` the periodic image of j moves with the
                # deforming lattice, so its velocity is offset.
                vrel_t = vi - (vj + dvj)
                vnnr = vrel_t.dot(dpos)
                vn_vec = dpos * (vnnr * rsqinv)
                vt_vec = vrel_t - vn_vec
                wr = (ri * oi + rj * oj) * rinv
                # LAMMPS: vtr1 = vt1 - (delz*wr2 - dely*wr3), i.e. vt - W x n.
                # With wr = W/r and dpos = r*n, dpos.cross(wr) = n x W = -(W x n),
                # so the rotational term is ADDED here.
                vtr = vt_vec + dpos.cross(wr)

                damp = meff * self.gamman * vnnr * rsqinv
                ccel = self.kn * delta * rinv - damp
                if self.limit_damping == 1 and ccel < 0.0:
                    ccel = 0.0

                # Tangential force (incremental shear history)
                jtag = tag[j]
                shear = shear_hist[nc]
                if partner_hist[nc] != jtag:
                    shear = ti.Vector([0.0, 0.0, 0.0])
                    partner_hist[nc] = jtag

                gammat_eff = meff * self.gammat

                # LAMMPS accumulates first, takes |shear|, and only then
                # projects the shear back into the tangent plane.
                if shearupdate != 0:
                    shear = shear + vtr * dt
                shrmag = shear.norm()
                if shearupdate != 0:
                    rsht = shear.dot(dpos) * rsqinv
                    shear = shear - rsht * dpos

                fs_vec = -(self.kt * shear + gammat_eff * vtr)
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
            else:
                # Not in contact: clear history
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
            atom.x,
            atom.v,
            atom.f,
            atom.omega,
            atom.torque,
            atom.radius,
            atom.rmass,
            atom.tag,
            atom.virial,
            nlist.npairs,
            nlist.pair_i,
            nlist.pair_j,
            history.shear,
            history.partner,
        )
