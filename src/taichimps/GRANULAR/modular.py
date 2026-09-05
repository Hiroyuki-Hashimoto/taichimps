"""
LAMMPS pair_style granular (modular sub-model framework) in Taichi.
Reference: LAMMPS src/GRANULAR/pair_granular.cpp
License: GPL v2 / taichimps MIT reimplementation

Supports combinations of:
  - normal: hooke, hertz
  - tangential: hooke_history, hertz_history, linear_history, off
  - damping: tsuji (normal), off
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.GRANULAR.base import GranularPair
from taichimps.neighbor import NeighborList


@ti.data_oriented
class GranularModular(GranularPair):
    """
    Modular granular pair style mimicking LAMMPS `pair_style granular`.
    """

    def __init__(
        self,
        domain: Domain,
        normal_model: str = "hooke",  # 'hooke' or 'hertz'
        tangential_model: str = "history",  # 'history' or 'none'
        kn: float = 1e5,
        gamman: float = 50.0,
        kt: float = 1e5,
        gammat: float = 50.0,
        xmu: float = 0.5,
        dampflag: int = 1,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)
        self.normal_model = normal_model
        self.tangential_model = tangential_model
        self.kn = float(kn)
        self.gamman = float(gamman)
        self.kt = float(kt)
        self.gammat = float(gammat)
        self.xmu = float(xmu)
        self.dampflag = dampflag

        self.is_hertz = 1 if normal_model.lower() == "hertz" else 0
        self.use_history = 1 if tangential_model.lower() in ("history", "hooke_history", "hertz_history") else 0

    @ti.kernel
    def compute_kernel(
        self,
        nlocal: ti.i32,
        dt: ti.f64,
        is_hertz: ti.i32,
        use_history: ti.i32,
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
                    n = dpos / r
                    delta = radsum - r

                    reff = (ri * rj) / (ri + rj)
                    meff = (mi * mj) / (mi + mj)
                    poly = 1.0
                    if is_hertz == 1:
                        poly = ti.sqrt(reff * delta)

                    w_cross_n = (ri * oi + rj * oj).cross(n)
                    vr = (vi - vj) - w_cross_n
                    vn = vr.dot(n)
                    vrn = vn * n
                    vrt = vr - vrn

                    kn_eff = self.kn * poly
                    gamman_eff = self.gamman * poly
                    if self.dampflag == 1:
                        gamman_eff *= meff

                    fn_elastic = kn_eff * delta
                    fn = fn_elastic - gamman_eff * vn
                    fn = max(fn, 0.0)

                    kt_eff = self.kt * poly
                    gammat_eff = self.gammat * poly
                    if self.dampflag == 1:
                        gammat_eff *= meff

                    ft_vec = ti.Vector([0.0, 0.0, 0.0])

                    if use_history == 1:
                        partner = partner_hist[i, k]
                        shear = shear_hist[i, k]
                        if partner != j:
                            shear = ti.Vector([0.0, 0.0, 0.0])
                            partner_hist[i, k] = j

                        shear_dot_n = shear.dot(n)
                        shear = shear - shear_dot_n * n
                        shear = shear + vrt * dt

                        ft_vec = -kt_eff * shear - gammat_eff * vrt
                        ft_mag = ft_vec.norm()
                        ft_max = self.xmu * fn_elastic

                        if ft_mag > ft_max and ft_mag > 1e-16:
                            scale = ft_max / ft_mag
                            ft_vec *= scale
                            shear = -ft_vec / kt_eff

                        shear_hist[i, k] = shear
                    else:
                        ft_vec = -gammat_eff * vrt
                        ft_mag = ft_vec.norm()
                        ft_max = self.xmu * fn_elastic
                        if ft_mag > ft_max and ft_mag > 1e-16:
                            ft_vec *= ft_max / ft_mag

                    f_total = fn * n + ft_vec
                    ti.atomic_add(f[i], f_total)
                    ti.atomic_add(f[j], -f_total)

                    n_cross_ft = n.cross(ft_vec)
                    ti.atomic_add(torque[i], -ri * n_cross_ft)
                    ti.atomic_add(torque[j], -rj * n_cross_ft)
                else:
                    if use_history == 1:
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
            self.is_hertz,
            self.use_history,
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
