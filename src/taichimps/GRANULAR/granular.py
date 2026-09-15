"""
LAMMPS pair_style granular (sub-model framework) in Taichi.

Reference: LAMMPS src/GRANULAR/pair_granular.cpp, granular_model.cpp and
           gran_sub_mod_{normal,damping,tangential}.cpp
License: GPL v2 / taichimps MIT reimplementation

Unlike the legacy `gran/*` styles, this one composes the contact law from
independently chosen normal, damping and tangential sub-models, each with its
own coefficients.  The subset implemented here covers what soil-mechanics
inputs actually use:

  normal      hooke | hertz | hertz/material
  damping     none | velocity | mass_velocity | viscoelastic | tsuji |
              coeff_restitution
  tangential  none | linear_nohistory | linear_history | mindlin

Anything else raises at construction rather than silently doing something
else.  Rolling, twisting, heat and the adhesive normal models (JKR, DMT, MDR)
are not implemented.

The per-contact sequence follows GranularModel::calculate_forces():

    delta = radsum - r,  Reff = ri*rj/radsum,  a = sqrt(delta * Reff)
    vtr   = vt - W x n              with W = ri*omega_i + rj*omega_j
    Fne   = normal_model(delta, a)
    Fdamp = -damp_prefactor * vnnr
    Fntot = Fne + Fdamp
    Fscrit = mu * |Fntot|           (the friction bound uses the TOTAL normal
                                     force, not just the elastic part)
    fs     = tangential_model(history, vtr, a), capped at Fscrit

The moment arm is `radi - delta/2` here, which is what pair granular uses and
is deliberately different from the legacy styles' `radi`.
"""

import math
from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.GRANULAR.base import GranularPair
from taichimps.neighbor import NeighborList

# Sub-model selectors. Resolved to ints at construction so the kernel can
# branch on them with ti.static() and compile the dead arms away.
NORMAL_HOOKE = 0
NORMAL_HERTZ = 1
NORMAL_HERTZ_MATERIAL = 2

DAMPING_NONE = 0
DAMPING_VELOCITY = 1
DAMPING_MASS_VELOCITY = 2
DAMPING_VISCOELASTIC = 3
DAMPING_TSUJI = 4
DAMPING_COEFF_RESTITUTION = 5

TANGENTIAL_NONE = 0
TANGENTIAL_LINEAR_NOHISTORY = 1
TANGENTIAL_LINEAR_HISTORY = 2
TANGENTIAL_MINDLIN = 3

_NORMAL_NAMES = {
    "hooke": NORMAL_HOOKE,
    "hertz": NORMAL_HERTZ,
    "hertz/material": NORMAL_HERTZ_MATERIAL,
}
_DAMPING_NAMES = {
    "none": DAMPING_NONE,
    "velocity": DAMPING_VELOCITY,
    "mass_velocity": DAMPING_MASS_VELOCITY,
    "viscoelastic": DAMPING_VISCOELASTIC,
    "tsuji": DAMPING_TSUJI,
    "coeff_restitution": DAMPING_COEFF_RESTITUTION,
}
_TANGENTIAL_NAMES = {
    "none": TANGENTIAL_NONE,
    "linear_nohistory": TANGENTIAL_LINEAR_NOHISTORY,
    "linear_history": TANGENTIAL_LINEAR_HISTORY,
    "mindlin": TANGENTIAL_MINDLIN,
}

# gran_sub_mod_tangential.cpp
EPSILON = 1e-10
# gran_sub_mod_damping.cpp
_TWOROOTFIVEBYSIX = 1.82574185835055380345  # 2*sqrt(5/6)
_ROOTTHREEBYTWO = 1.22474487139158894067  # sqrt(3/2)
_FOURTHIRDS = 4.0 / 3.0


def mix_stiffness_e(e1: float, e2: float, poiss1: float, poiss2: float) -> float:
    """GranSubMod::mix_stiffnessE"""
    return 1.0 / ((1 - poiss1 * poiss1) / e1 + (1 - poiss2 * poiss2) / e2)


def mix_stiffness_g(e1: float, e2: float, poiss1: float, poiss2: float) -> float:
    """GranSubMod::mix_stiffnessG"""
    return 1.0 / (
        2 * (2 - poiss1) * (1 + poiss1) / e1 + 2 * (2 - poiss2) * (1 + poiss2) / e2
    )


@ti.data_oriented
class PairGranular(GranularPair):
    """
    pair_style granular with a selectable normal / damping / tangential model.

    Coefficients follow the LAMMPS pair_coeff ordering:
      normal hooke            k damp
      normal hertz            k damp
      normal hertz/material   E damp poisson
      tangential linear_nohistory  xt mu          (no stiffness)
      tangential linear_history    k|NULL xt mu
      tangential mindlin           k|NULL xt mu   (NULL derives k from E, nu)
    For `damping coeff_restitution` and `tsuji`, the normal model's `damp`
    coefficient is the coefficient of restitution.
    """

    def __init__(
        self,
        domain: Domain,
        normal: str,
        normal_coeffs: list[float],
        tangential: str = "none",
        tangential_coeffs: list[float | None] | None = None,
        damping: str = "viscoelastic",
        limit_damping: bool = False,
        float_type: Any = ti.f64,
    ) -> None:
        super().__init__(domain, float_type=float_type)

        if normal not in _NORMAL_NAMES:
            raise ValueError(
                f"Unsupported pair granular normal model {normal!r}; "
                f"implemented: {sorted(_NORMAL_NAMES)}"
            )
        if damping not in _DAMPING_NAMES:
            raise ValueError(
                f"Unsupported pair granular damping model {damping!r}; "
                f"implemented: {sorted(_DAMPING_NAMES)}"
            )
        if tangential not in _TANGENTIAL_NAMES:
            raise ValueError(
                f"Unsupported pair granular tangential model {tangential!r}; "
                f"implemented: {sorted(_TANGENTIAL_NAMES)}"
            )

        self.normal_id = _NORMAL_NAMES[normal]
        self.damping_id = _DAMPING_NAMES[damping]
        self.tangential_id = _TANGENTIAL_NAMES[tangential]
        self.limit_damping = 1 if limit_damping else 0

        self.emod = 0.0
        self.poiss = 0.0
        if self.normal_id == NORMAL_HERTZ_MATERIAL:
            if len(normal_coeffs) != 3:
                raise ValueError("normal hertz/material takes E, damp, poisson")
            self.emod, cor, self.poiss = (float(c) for c in normal_coeffs)
            # GranSubModNormalHertzMaterial::coeffs_to_local
            self.kn = _FOURTHIRDS * mix_stiffness_e(
                self.emod, self.emod, self.poiss, self.poiss
            )
        else:
            if len(normal_coeffs) != 2:
                raise ValueError(f"normal {normal} takes k, damp")
            self.kn, cor = (float(c) for c in normal_coeffs)
        self.normal_damp = float(cor)

        self.damp = self._damping_coefficient(normal)

        # `contact_radius` only enters the normal force for the Hertz variants;
        # GranSubModNormalHooke::calculate_forces() is just k * delta.
        self.normal_uses_contact_radius = self.normal_id != NORMAL_HOOKE

        self.kt = 0.0
        self.xt = 0.0
        self.mu = 0.0
        # mindlin scales the tangential stiffness by the contact radius;
        # linear_history is otherwise identical but does not.
        self.tangential_scales_k = self.tangential_id == TANGENTIAL_MINDLIN
        self.uses_history = self.tangential_id in (
            TANGENTIAL_LINEAR_HISTORY,
            TANGENTIAL_MINDLIN,
        )
        if self.tangential_id != TANGENTIAL_NONE:
            self._setup_tangential(tangential, tangential_coeffs or [])

        # The pair_coeff arguments, verbatim.  A restart file stores the model
        # by sub-model name and raw coefficients (GranularModel::write_restart),
        # not by the derived stiffnesses above, so they have to be kept.  NULL
        # is stored as the -1 sentinel LAMMPS uses.
        self.submodels: dict[str, tuple[str, list[float]]] = {
            "normal": (normal, [float(c) for c in normal_coeffs]),
            "damping": (damping, []),
            "tangential": (
                tangential,
                [-1.0 if c is None else float(c) for c in (tangential_coeffs or [])],
            ),
            "rolling": ("none", []),
            "twisting": ("none", []),
            "heat": ("none", []),
        }

    def _damping_coefficient(self, normal: str) -> float:
        """The `damp` constant of the chosen damping sub-model (its init())."""
        if self.damping_id in (DAMPING_TSUJI, DAMPING_COEFF_RESTITUTION):
            cor = self.normal_damp
            if not 0.0 < cor <= 1.0:
                raise ValueError(
                    "damping tsuji/coeff_restitution needs the normal model's "
                    f"damp coefficient to be a restitution coefficient in (0, 1]; got {cor}"
                )
            logcor = math.log(cor)
            if self.damping_id == DAMPING_TSUJI:
                # Eq. 53 of Marshall 2009, times sqrt(2) for the meff convention
                d = (
                    1.2728
                    - 4.2783 * cor
                    + 11.087 * cor**2
                    - 22.348 * cor**3
                    + 27.467 * cor**4
                    - 18.022 * cor**5
                    + 4.8218 * cor**6
                )
                return d * math.sqrt(2.0)
            # coeff_restitution: Hookean and Hertzian forms differ
            if normal == "hooke":
                return -2 * logcor / math.sqrt(math.pi**2 + logcor**2)
            return (
                -_ROOTTHREEBYTWO
                * _TWOROOTFIVEBYSIX
                * logcor
                / math.sqrt(math.pi**2 + logcor**2)
            )
        return self.normal_damp

    def _setup_tangential(self, tangential: str, coeffs: list[float | None]) -> None:
        if self.tangential_id == TANGENTIAL_LINEAR_NOHISTORY:
            if len(coeffs) != 2 or any(c is None for c in coeffs):
                raise ValueError("tangential linear_nohistory takes xt, mu")
            self.xt, self.mu = (float(c) for c in coeffs)  # type: ignore[arg-type]
            return

        if len(coeffs) != 3:
            raise ValueError(f"tangential {tangential} takes k (or NULL), xt, mu")
        k_raw, xt, mu = coeffs
        if xt is None or mu is None:
            raise ValueError(f"tangential {tangential}: xt and mu cannot be NULL")
        self.xt = float(xt)
        self.mu = float(mu)

        if k_raw is None:
            # NULL: derive from the normal model's material properties.
            if self.normal_id != NORMAL_HERTZ_MATERIAL:
                raise ValueError(
                    "a NULL tangential stiffness requires normal hertz/material, "
                    "which is where E and Poisson's ratio come from"
                )
            if self.tangential_id == TANGENTIAL_MINDLIN:
                self.kt = 8.0 * mix_stiffness_g(
                    self.emod, self.emod, self.poiss, self.poiss
                )
            else:
                raise ValueError("tangential linear_history needs an explicit stiffness")
        else:
            self.kt = float(k_raw)

    @ti.func
    def _rotate_rescale(self, vec, n):
        """GranSubMod::rotate_rescale_vec: project onto the tangent plane, keep |vec|."""
        shrmag = vec.norm()
        out = vec - vec.dot(n) * n
        prjmag = out.norm()
        scale = 0.0
        if prjmag > 0.0:
            scale = shrmag / prjmag
        return out * scale

    @ti.kernel
    def compute_kernel(
        self,
        nlocal: ti.template(),
        dt: ti.template(),
        history_update: ti.i32,
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
        # `nlocal` is a ti.template(), i.e. baked into the compiled kernel,
        # not passed at launch. A loop whose bound is a runtime argument makes
        # Taichi emit a second, serial GPU launch ahead of the parallel one
        # just to establish the range; measured at 6.9 us per call on CUDA,
        # which at this system size is a third of the kernel's own cost.
        # Taichi recompiles per distinct value, so a run whose particle count
        # never changes compiles this once.
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

                rj = radius[j]
                radsum = ri + rj
                dpos, dvj = self.domain.minimum_image_and_vshift(xi - x[j])
                rsq = dpos.dot(dpos)

                if rsq < radsum * radsum and rsq > 1e-28:
                    r = ti.sqrt(rsq)
                    rinv = 1.0 / r
                    n = dpos * rinv
                    delta = radsum - r
                    reff = (ri * rj) / radsum
                    meff = (mi * rmass[j]) / (mi + rmass[j])
                    # contact_radius = sqrt(dR), dR = delta * Reff
                    a = ti.sqrt(delta * reff)

                    # Under `remap v` the periodic image of j moves with the
                    # deforming lattice, so its velocity is offset.
                    vr = vi - (v[j] + dvj)
                    vnnr = vr.dot(n)
                    vt = vr - vnnr * n
                    wr = ri * oi + rj * omega[j]
                    # GranularModel: cross3(wr, nx, temp); sub3(vt, temp, vtr)
                    vtr = vt - wr.cross(n)
                    vrel = vtr.norm()

                    # --- normal elastic force -------------------------------
                    fne = 0.0
                    if ti.static(self.normal_uses_contact_radius):
                        fne = self.kn * a * delta
                    else:
                        fne = self.kn * delta

                    # --- damping --------------------------------------------
                    damp_prefactor = 0.0
                    if ti.static(self.damping_id == DAMPING_VELOCITY):
                        damp_prefactor = self.damp
                    elif ti.static(self.damping_id == DAMPING_MASS_VELOCITY):
                        damp_prefactor = self.damp * meff
                    elif ti.static(self.damping_id == DAMPING_VISCOELASTIC):
                        damp_prefactor = self.damp * meff * a
                    elif ti.static(
                        self.damping_id in (DAMPING_TSUJI, DAMPING_COEFF_RESTITUTION)
                    ):
                        arg = 0.0
                        if delta > 0.0:
                            arg = ti.max(0.0, meff * fne / delta)
                        damp_prefactor = self.damp * ti.sqrt(arg)

                    fntot = fne - damp_prefactor * vnnr
                    if ti.static(self.limit_damping == 1):
                        fntot = max(fntot, 0.0)

                    # GranSubModNormal::set_fncrit()
                    fscrit = self.mu * ti.abs(fntot)

                    # --- tangential -----------------------------------------
                    fs = ti.Vector([0.0, 0.0, 0.0])
                    if ti.static(self.tangential_id == TANGENTIAL_LINEAR_NOHISTORY):
                        damp_t = self.xt * damp_prefactor
                        ft = 0.0
                        if vrel != 0.0:
                            ft = ti.min(fscrit, damp_t * vrel) / vrel
                        fs = -ft * vtr
                    elif ti.static(self.uses_history):
                        jtag = tag[j]
                        hist = shear_hist[i, k]
                        if partner_hist[i, k] != jtag:
                            hist = ti.Vector([0.0, 0.0, 0.0])
                            partner_hist[i, k] = jtag

                        k_eff = self.kt
                        if ti.static(self.tangential_scales_k):
                            k_eff = self.kt * a
                        damp_t = self.xt * damp_prefactor

                        if history_update != 0:
                            rsht = hist.dot(n)
                            if ti.abs(rsht) * k_eff > EPSILON * fscrit:
                                hist = self._rotate_rescale(hist, n)
                            hist = hist + vtr * dt

                        fdamp = -damp_t * vtr
                        fs = -k_eff * hist + fdamp

                        magfs = fs.norm()
                        if magfs > fscrit:
                            if hist.norm() != 0.0:
                                fs = fs * (fscrit / magfs)
                                # history = elastic part of the rescaled force
                                hist = -(fs - fdamp) / k_eff
                            else:
                                fs = ti.Vector([0.0, 0.0, 0.0])

                        shear_hist[i, k] = hist

                    f_total = fntot * n + fs

                    ti.atomic_add(f[i], f_total)
                    ti.atomic_add(f[j], -f_total)

                    # torquesi/j = -(n x fs) * (rad - delta/2)
                    tor = -n.cross(fs)
                    ti.atomic_add(torque[i], (ri - 0.5 * delta) * tor)
                    ti.atomic_add(torque[j], (rj - 0.5 * delta) * tor)

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
                    if ti.static(self.uses_history):
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
