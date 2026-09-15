"""
Box deformation with optional pressure servo control.

Reference: LAMMPS src/EXTRA-FIX/fix_deform_pressure.cpp and src/fix_deform.cpp
License: GPL v2 compatible / MIT reimplementation

Each of x, y, z and each tilt factor xy, xz, yz carries its own control style,
and strain-controlled and pressure-controlled entries can be mixed -- which is
what makes the standard triaxial and simple-shear setups expressible in a single
fix:

    fix 1 all deform/pressure 1 z trate -0.01 \\
        x pressure 100000.0 1e-4 y pressure 100000.0 1e-4 max/rate 0.1
    fix 1 all deform/pressure 1 xy erate 0.001

Implemented styles
    x/y/z     none | erate | trate | vel | wiggle | variable |
              final | scale | delta | volume | pressure | pressure/mean
    xy/xz/yz  none | erate | trate | vel | wiggle | variable |
              final | delta | pressure
    box       volume | pressure
    keywords  couple, max/rate, normalize/pressure, vol/balance/p, remap, nevery

`remap x` (the default) moves the atoms affinely with the box.  `remap v`
instead leaves coordinates alone and corrects the velocity of an atom that
crosses a periodic boundary by the box deformation rate, which is carried in
Domain.h_rate; the deformation then reaches the assembly through contacts
rather than through an imposed displacement field.

Not implemented: `erate/rescale` (it exists for NEMD with fix nvt/sllod, which
taichimps does not have) and `units lattice` (there is no lattice command).
Both raise rather than being silently ignored.

Entries are indexed the way LAMMPS indexes them, 0..5 = x, y, z, yz, xz, xy, so
that the ordering-sensitive parts (the {5, 3, 4} tilt sweep, the flip bookkeeping
where a yz flip moves xz) can be read against the original.

Box updates follow the LAMMPS bookkeeping: targets are always expressed relative
to the box as it was at the start of the run (`lo_start`/`hi_start`/`tilt_start`)
plus a `cumulative_shift`, rather than as an increment on the current box, so the
trajectory does not depend on rounding of intermediate states.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.computes.thermo import Computes
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix

# LAMMPS Set indices.
IX, IY, IZ, IYZ, IXZ, IXY = 0, 1, 2, 3, 4, 5
DIM_INDEX = {"x": IX, "y": IY, "z": IZ}
TILT_INDEX = {"yz": IYZ, "xz": IXZ, "xy": IXY}
# Domain.tilt is stored as [xy, xz, yz]; Set indices 5, 4, 3 map onto it.
TILT_TO_DOMAIN = {IXY: 0, IXZ: 1, IYZ: 2}

STRAIN_STYLES = ("erate", "trate", "vel", "wiggle", "variable", "final", "scale", "delta")
PRESSURE_STYLES = ("pressure", "pressure/mean")
DIM_STYLES = ("none", "volume", *STRAIN_STYLES, *PRESSURE_STYLES)
# `scale` and `volume` are box-size concepts with no tilt analogue in LAMMPS.
TILT_STYLES = ("none", "erate", "trate", "vel", "wiggle", "variable", "final", "delta",
               "pressure")
BOX_STYLES = ("none", "volume", "pressure")

COUPLE_CHOICES = ("none", "xyz", "xy", "yz", "xz")

# Styles whose target is interpolated between the box at run start and a box at
# run end, so they need to know how long the run is.
NEEDS_RUN_LENGTH = ("final", "scale", "delta")


@dataclass
class AxisSet:
    """Per-dimension or per-tilt control settings, the equivalent of LAMMPS `Set`."""

    style: str = "none"
    # strain parameters
    rate: float = 0.0          # erate / trate
    vel: float = 0.0           # vel
    scale: float = 1.0         # scale
    flo: float = 0.0           # final (box dims)
    fhi: float = 0.0
    dlo: float = 0.0           # delta (box dims)
    dhi: float = 0.0
    ftilt: float = 0.0         # final (tilt)
    dtilt: float = 0.0         # delta (tilt)
    amplitude: float = 0.0     # wiggle
    tperiod: float = 0.0
    hstr: str = ""             # variable: box length / tilt change
    hratestr: str = ""         # variable: rate of change
    # pressure parameters
    ptarget: float = 0.0
    pgain: float = 0.0
    # run-time state
    lo_start: float = 0.0
    hi_start: float = 0.0
    lo_target: float = 0.0
    hi_target: float = 0.0
    tilt_start: float = 0.0
    tilt_target: float = 0.0
    cumulative_shift: float = 0.0
    vol_start: float = 0.0
    # vol/balance/p state, LAMMPS SetExtra
    prior_rate: float = 0.0
    prior_pressure: float = 0.0
    saved: bool = False
    # volume coupling
    substyle: str = ""
    fixed: int = -1
    dynamic1: int = -1
    dynamic2: int = -1
    coupled: bool = False
    # `box` style only
    cumulative_vshift: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@ti.data_oriented
class FixDeformPressure(Fix):
    """
    Deform the periodic box, optionally servo-controlling it to a target pressure.

    A pressure-controlled entry uses proportional control on the engineering
    strain rate, as in FixDeformPressure::apply_pressure():

        strain_rate = pgain * (p_current - p_target)

    `normalize_pressure` divides that by |p_target| so the gain becomes
    dimensionless, and `max_rate` caps the magnitude.  `pressure` follows the
    matching component of the pressure tensor, `pressure/mean` follows the
    scalar mean pressure.
    """

    def __init__(
        self,
        domain: Domain,
        axes: dict[str, dict[str, Any]] | None = None,
        box: dict[str, Any] | None = None,
        couple: str = "none",
        max_rate: float = 0.0,
        normalize_pressure: bool = False,
        vol_balance_p: bool = False,
        remap: str = "x",
        nevery: int = 1,
        flip: bool = True,
        var_eval: Callable[[str], float] | None = None,
        float_type: Any = ti.f64,
        # Legacy single-target isotropic form, kept so existing callers and
        # scripts that only asked for one confining pressure keep working.
        p_target: float | None = None,
        p_gain: float | None = None,
    ) -> None:
        super().__init__(domain, float_type)
        self.nevery = int(nevery)
        if self.nevery < 1:
            raise ValueError("fix deform/pressure nevery must be >= 1")

        if couple not in COUPLE_CHOICES:
            raise ValueError(f"couple must be one of {COUPLE_CHOICES}, got {couple!r}")
        if remap not in ("x", "v", "none"):
            raise ValueError("remap must be 'x', 'v' or 'none'")
        self.couple = couple
        self.remap = remap
        self.flip = bool(flip)
        self.max_rate = float(max_rate)
        self.normalize_pressure = bool(normalize_pressure)
        self.vol_balance_p = bool(vol_balance_p)
        self.var_eval = var_eval

        self.sets = [AxisSet() for _ in range(6)]
        self.set_box = AxisSet()

        if axes is None and box is None:
            # Isotropic pressure control on all three axes.
            target = 50000.0 if p_target is None else float(p_target)
            gain = 1e-4 if p_gain is None else float(p_gain)
            for s in self.sets[:3]:
                s.style = "pressure"
                s.ptarget = target
                s.pgain = gain
            if couple == "none":
                self.couple = "xyz"
        else:
            self._configure(axes or {}, box)

        self._validate()

        # Scratch fields for the remap kernel. Held as fields rather than
        # passed as scalars so the kernel is compiled once regardless of
        # float_type and of how often the values change.
        self._old_lo = ti.Vector.field(3, dtype=float_type, shape=())
        self._new_lo = ti.Vector.field(3, dtype=float_type, shape=())
        # h as [xprd, yprd, zprd] plus [xy, xz, yz], before and after.
        self._old_prd = ti.Vector.field(3, dtype=float_type, shape=())
        self._old_hinv_tilt = ti.Vector.field(3, dtype=float_type, shape=())
        self._new_prd = ti.Vector.field(3, dtype=float_type, shape=())
        self._new_tilt = ti.Vector.field(3, dtype=float_type, shape=())

        self.computes = Computes(float_type=float_type)
        self.step_count = 0
        self.nsteps = 0
        self.nsteps_total = 0
        self._started = False

    # ------------------------------------------------------------------ setup

    def _configure(self, axes: dict[str, dict[str, Any]], box: dict[str, Any] | None) -> None:
        for name, spec in axes.items():
            allowed: tuple[str, ...]
            if name in DIM_INDEX:
                idx, allowed = DIM_INDEX[name], DIM_STYLES
            elif name in TILT_INDEX:
                idx, allowed = TILT_INDEX[name], TILT_STYLES
            else:
                raise ValueError(f"Unknown deform dimension {name!r}")

            style = spec["style"]
            if style == "erate/rescale":
                raise ValueError(
                    "fix deform erate/rescale exists for NEMD with fix nvt/sllod, "
                    "which taichimps does not have"
                )
            if style not in allowed:
                raise ValueError(
                    f"Unsupported fix deform/pressure style {style!r} for {name}; "
                    f"implemented: {allowed}"
                )
            s = self.sets[idx]
            s.style = style
            for key, value in spec.items():
                if key == "style":
                    continue
                if not hasattr(s, key):
                    raise ValueError(f"Unknown parameter {key!r} for dimension {name}")
                setattr(s, key, value if isinstance(value, str) else float(value))

        if box is not None:
            style = box["style"]
            if style not in BOX_STYLES:
                raise ValueError(
                    f"Unsupported fix deform/pressure box style {style!r}; "
                    f"implemented: {BOX_STYLES}"
                )
            self.set_box.style = style
            for key, value in box.items():
                if key != "style":
                    setattr(self.set_box, key, float(value))

    def _validate(self) -> None:
        names = ["x", "y", "z", "yz", "xz", "xy"]
        for i, s in enumerate(self.sets):
            if s.style in PRESSURE_STYLES and s.pgain <= 0.0:
                raise ValueError(
                    f"fix deform/pressure gain for {names[i]} must be positive"
                )
            if s.style == "variable" and not (s.hstr and s.hratestr):
                raise ValueError(
                    f"fix deform/pressure variable style for {names[i]} needs two "
                    "variable names"
                )
            if s.style == "wiggle" and s.tperiod <= 0.0:
                raise ValueError(f"fix deform/pressure wiggle for {names[i]} needs a period")

        if any(s.style == "variable" for s in self.sets) and self.var_eval is None:
            raise ValueError(
                "fix deform/pressure variable style needs a var_eval callback to "
                "evaluate the equal-style variables"
            )

        has_pressure = any(s.style in PRESSURE_STYLES for s in self.sets)
        has_pressure = has_pressure or self.set_box.style == "pressure"
        if self.normalize_pressure and not has_pressure:
            raise ValueError(
                "normalize/pressure only applies to pressure-controlled dimensions"
            )
        if self.max_rate < 0.0:
            raise ValueError("max/rate must be positive")

        volume_dims = [i for i in range(3) if self.sets[i].style == "volume"]
        if volume_dims and any(s.style == "pressure/mean" for s in self.sets):
            raise ValueError(
                "cannot ask fix deform/pressure for constant volume and a mean "
                "pressure at the same time"
            )
        if self.set_box.style != "none":
            bad = ("final", "delta", "scale", "pressure/mean", "variable", "volume")
            for i in range(3):
                if self.sets[i].style in bad:
                    raise ValueError(
                        "fix deform/pressure box may only be combined with x/y/z "
                        f"styles vel, erate, trate, wiggle and pressure, not "
                        f"{self.sets[i].style!r}"
                    )
        if self.vol_balance_p and not volume_dims:
            raise ValueError("vol/balance/p only applies with the volume style")

        if self.couple != "none":
            idx = [i for i, c in enumerate("xyz") if c in self.couple]
            ref = next((i for i in idx if self.sets[i].style in PRESSURE_STYLES), None)
            if ref is None:
                raise ValueError(
                    "couple needs at least one of the coupled dimensions to be "
                    "pressure controlled"
                )
            for i in idx:
                s = self.sets[i]
                s.coupled = True
                if s.style == "none":
                    # LAMMPS copies the reference dimension's control over.
                    s.style = self.sets[ref].style
                    s.ptarget = self.sets[ref].ptarget
                    s.pgain = self.sets[ref].pgain
                elif s.style not in PRESSURE_STYLES:
                    raise ValueError("cannot couple non-pressure-controlled dimensions")
                elif (
                    s.pgain != self.sets[ref].pgain
                    or s.ptarget != self.sets[ref].ptarget
                ):
                    raise ValueError(
                        "coupled dimensions must have identical target and gain"
                    )

        self._setup_volume_links()

    def _setup_volume_links(self) -> None:
        """Resolve which dimensions a `volume` axis derives its size from."""
        for i in range(3):
            s = self.sets[i]
            if s.style != "volume":
                continue
            o1, o2 = (i + 1) % 3, (i + 2) % 3
            s1, s2 = self.sets[o1].style, self.sets[o2].style
            if s1 in ("none", "volume") and s2 in ("none", "volume"):
                raise ValueError(
                    "the volume style needs at least one other deformed dimension"
                )
            if s1 == "none":
                s.substyle, s.fixed, s.dynamic1 = "one_from_one", o1, o2
            elif s2 == "none":
                s.substyle, s.fixed, s.dynamic1 = "one_from_one", o2, o1
            elif s1 == "volume":
                s.substyle, s.fixed, s.dynamic1 = "two_from_one", o1, o2
            elif s2 == "volume":
                s.substyle, s.fixed, s.dynamic1 = "two_from_one", o2, o1
            else:
                s.substyle, s.dynamic1, s.dynamic2 = "one_from_two", o1, o2

            if self.vol_balance_p and s.substyle != "two_from_one":
                raise ValueError(
                    "vol/balance/p needs two dimensions holding the volume constant"
                )

    def setup(self, nsteps_total: int = 0, dt: float = 0.0) -> None:
        """
        Latch the box at the start of a run, the way FixDeform::init() does.

        `nsteps_total` is only needed by the styles that interpolate towards a
        box at the end of the run (final, scale, delta).
        """
        assert self.domain is not None
        self.nsteps = 0
        self.nsteps_total = int(nsteps_total)
        self._started = True
        vol = float(self.domain.volume)

        for i in range(3):
            s = self.sets[i]
            s.lo_start = float(self.domain.boxlo[i])
            s.hi_start = float(self.domain.boxhi[i])
            s.lo_target, s.hi_target = s.lo_start, s.hi_start
            s.cumulative_shift = 0.0
            s.vol_start = vol
            s.saved = False
        for i in (IYZ, IXZ, IXY):
            s = self.sets[i]
            s.tilt_start = float(self.domain.tilt[TILT_TO_DOMAIN[i]])
            s.tilt_target = s.tilt_start
            s.cumulative_shift = 0.0
        self.set_box.vol_start = vol
        self.set_box.cumulative_vshift = [0.0, 0.0, 0.0]

        for i, s in enumerate(self.sets):
            if s.style in NEEDS_RUN_LENGTH and self.nsteps_total <= 0:
                raise ValueError(
                    f"fix deform/pressure style {s.style!r} needs the run length; "
                    "call setup(nsteps_total) before running"
                )

        # FixDeform::init() fills h_rate for the strain styles before the run
        # starts, so `remap v` already has a rate to apply at the very first
        # boundary wrap. Leaving it at zero until the first end_of_step makes
        # the first step disagree with LAMMPS.
        self.domain.set_h_rate(self._strain_h_rate(dt), vremap=(self.remap == "v"))

    def _strain_h_rate(self, dt: float) -> np.ndarray:
        """
        Analytic box-length rates of the strain styles, as FixDeform::init().

        Pressure- and volume-controlled entries contribute nothing here; their
        rate only exists once a pressure has been measured.
        """
        h = np.zeros(6)
        delt = self.nsteps_total * dt
        for i in range(3):
            s = self.sets[i]
            length = s.hi_start - s.lo_start
            if s.style in ("erate", "trate"):
                h[i] = s.rate * length
            elif s.style == "vel":
                h[i] = s.vel
            elif s.style == "wiggle":
                h[i] = 2.0 * math.pi / s.tperiod * s.amplitude
            elif s.style in NEEDS_RUN_LENGTH and delt > 0.0:
                if s.style == "final":
                    stop = s.fhi - s.flo
                elif s.style == "delta":
                    stop = length + s.dhi - s.dlo
                else:
                    stop = s.scale * length
                h[i] = (stop - length) / delt
        # h_rate order past the diagonal is [yz, xz, xy]
        for slot, idx in ((3, IYZ), (4, IXZ), (5, IXY)):
            s = self.sets[idx]
            b = IY if idx == IXY else IZ
            lb = self.sets[b].hi_start - self.sets[b].lo_start
            if s.style == "erate":
                h[slot] = s.rate * lb
            elif s.style == "trate":
                h[slot] = s.rate * s.tilt_start
            elif s.style == "vel":
                h[slot] = s.vel
            elif s.style == "wiggle":
                h[slot] = 2.0 * math.pi / s.tperiod * s.amplitude
            elif s.style in ("final", "delta") and delt > 0.0:
                stop = s.ftilt if s.style == "final" else s.tilt_start + s.dtilt
                h[slot] = (stop - s.tilt_start) / delt
        return h

    # ------------------------------------------------------------------ kernel

    @ti.kernel
    def remap_positions(self, nlocal: ti.i32, x: ti.template()):
        """
        Affine remap, the lamda-coordinate round trip LAMMPS performs.

        x is converted to fractional coordinates with the *old* shape matrix and
        back with the *new* one, so it follows a change of tilt as well as of
        box length.  Anchoring on the new lower bound matters: anchoring on the
        old one (as this used to) leaves the particles displaced from the box by
        half the box-length change on every single update.
        """
        old_lo = self._old_lo[None]
        new_lo = self._new_lo[None]
        oprd = self._old_prd[None]
        ohinv = self._old_hinv_tilt[None]  # [h_inv5, h_inv4, h_inv3]
        nprd = self._new_prd[None]
        nt = self._new_tilt[None]          # [xy, xz, yz]

        for i in range(nlocal):
            d = x[i] - old_lo
            lam = ti.Vector([
                d[0] / oprd[0] + ohinv[0] * d[1] + ohinv[1] * d[2],
                d[1] / oprd[1] + ohinv[2] * d[2],
                d[2] / oprd[2],
            ])
            x[i] = ti.Vector([
                nprd[0] * lam[0] + nt[0] * lam[1] + nt[1] * lam[2] + new_lo[0],
                nprd[1] * lam[1] + nt[2] * lam[2] + new_lo[1],
                nprd[2] * lam[2] + new_lo[2],
            ])

    # ------------------------------------------------------------------- steps

    def _current_pressure(self, atom: AtomSystem) -> tuple[np.ndarray, np.ndarray]:
        """
        Pressure to servo on, per Set entry, after applying the couple setting.

        Returns (per-entry target pressure, raw tensor) so the box style and the
        volume balancer can reuse the tensor.
        """
        assert self.domain is not None
        tensor = self.computes.compute_pressure_tensor(atom, self.domain)
        scalar = float(np.mean(tensor[:3]))

        p = np.zeros(6, dtype=np.float64)
        if self.couple == "xyz":
            p[:3] = np.mean(tensor[:3])
        elif self.couple in ("xy", "yz", "xz"):
            idx = [i for i, c in enumerate("xyz") if c in self.couple]
            other = [i for i in range(3) if i not in idx]
            p[idx] = np.mean(tensor[idx])
            for i in other:
                p[i] = tensor[i]
        else:
            for i in range(3):
                if self.sets[i].style == "pressure":
                    p[i] = tensor[i]
                elif self.sets[i].style == "pressure/mean":
                    p[i] = scalar
        # Tilt entries: Set 3 (yz) follows tensor[5], 4 (xz) tensor[4],
        # 5 (xy) tensor[3], in the [xx, yy, zz, xy, xz, yz] ordering.
        p[IYZ], p[IXZ], p[IXY] = tensor[5], tensor[4], tensor[3]
        return p, tensor

    def _limit_rate(self, rate: float, ptarget: float) -> float:
        """normalize/pressure then max/rate, in the LAMMPS order."""
        if self.normalize_pressure:
            if ptarget == 0.0:
                if self.max_rate == 0.0:
                    raise ValueError(
                        "cannot normalize the error for a zero target pressure "
                        "without a max/rate"
                    )
                if rate != 0.0:
                    rate = math.copysign(self.max_rate, rate)
            else:
                rate /= abs(ptarget)
        if self.max_rate != 0.0 and abs(rate) > self.max_rate:
            rate = math.copysign(self.max_rate, rate)
        return rate

    def _var(self, name: str) -> float:
        assert self.var_eval is not None
        return float(self.var_eval(name))

    def _apply_strain(self, dt: float) -> None:
        """Set targets for the strain-controlled dimensions and tilt factors."""
        assert self.domain is not None
        # `self` is a @ti.data_oriented object, so every attribute access on it
        # goes through Taichi's __getattribute__ hook -- about 0.7 us each, and
        # this runs every step. Bind what the loops need to locals first.
        dom = self.domain
        sets = self.sets
        elapsed = self.nsteps * dt
        frac = self.nsteps / self.nsteps_total if self.nsteps_total > 0 else 0.0
        boxlo, boxhi, dom_tilt = dom.boxlo, dom.boxhi, dom.tilt

        for i in range(3):
            s = sets[i]
            if s.style == "none":
                s.lo_target = float(boxlo[i])
                s.hi_target = float(boxhi[i])
            elif s.style == "trate":
                # True strain rate: L(t) = L0 * exp(rate * t)
                mid = 0.5 * (s.lo_start + s.hi_start)
                half = 0.5 * (s.hi_start - s.lo_start) * math.exp(s.rate * elapsed)
                s.lo_target, s.hi_target = mid - half, mid + half
            elif s.style == "erate":
                # Engineering strain rate: L(t) = L0 * (1 + rate * t)
                shift = 0.5 * elapsed * s.rate * (s.hi_start - s.lo_start)
                s.lo_target, s.hi_target = s.lo_start - shift, s.hi_start + shift
            elif s.style == "vel":
                shift = 0.5 * elapsed * s.vel
                s.lo_target, s.hi_target = s.lo_start - shift, s.hi_start + shift
            elif s.style == "wiggle":
                shift = 0.5 * s.amplitude * math.sin(2.0 * math.pi * elapsed / s.tperiod)
                s.lo_target, s.hi_target = s.lo_start - shift, s.hi_start + shift
            elif s.style == "variable":
                delta_l = self._var(s.hstr)
                s.lo_target = s.lo_start - 0.5 * delta_l
                s.hi_target = s.hi_start + 0.5 * delta_l
            elif s.style in NEEDS_RUN_LENGTH:
                if s.style == "final":
                    lo_stop, hi_stop = s.flo, s.fhi
                elif s.style == "delta":
                    lo_stop = s.lo_start + s.dlo
                    hi_stop = s.hi_start + s.dhi
                else:  # scale
                    mid = 0.5 * (s.lo_start + s.hi_start)
                    half = 0.5 * s.scale * (s.hi_start - s.lo_start)
                    lo_stop, hi_stop = mid - half, mid + half
                s.lo_target = s.lo_start + frac * (lo_stop - s.lo_start)
                s.hi_target = s.hi_start + frac * (hi_stop - s.hi_start)

        # Tilt factors, in LAMMPS's order: xy first, then yz, then xz, since xz
        # depends on the xy rate and the yz tilt.
        for i in (IXY, IYZ, IXZ):
            s = sets[i]
            # `b` is the dimension the shear rate is measured against: y for xy,
            # z for xz and yz.
            b = IY if i == IXY else IZ
            lb_start = sets[b].hi_start - sets[b].lo_start
            if s.style == "none":
                s.tilt_target = float(dom_tilt[TILT_TO_DOMAIN[i]])
            elif s.style == "trate":
                s.tilt_target = s.tilt_start * math.exp(s.rate * elapsed)
            elif s.style == "erate":
                s.tilt_target = s.tilt_start + elapsed * s.rate * lb_start
            elif s.style == "vel":
                s.tilt_target = s.tilt_start + elapsed * s.vel
            elif s.style == "wiggle":
                s.tilt_target = s.tilt_start + s.amplitude * math.sin(
                    2.0 * math.pi * elapsed / s.tperiod
                )
            elif s.style == "variable":
                s.tilt_target = s.tilt_start + self._var(s.hstr)
            elif s.style in ("final", "delta"):
                stop = s.ftilt if s.style == "final" else s.tilt_start + s.dtilt
                s.tilt_target = s.tilt_start + frac * (stop - s.tilt_start)

    def _apply_pressure(self, p_current: np.ndarray, dt: float) -> None:
        """Proportional servo on the pressure-controlled entries."""
        assert self.domain is not None
        sets = self.sets
        prd = self.domain.prd
        limit = self._limit_rate
        for i in range(3):
            s = sets[i]
            if s.style not in PRESSURE_STYLES:
                continue
            rate = limit(s.pgain * (p_current[i] - s.ptarget), s.ptarget)
            # h_rate is a rate of change of box length, not a strain rate
            s.cumulative_shift += dt * rate * float(prd[i])
            s.lo_target = s.lo_start - 0.5 * s.cumulative_shift
            s.hi_target = s.hi_start + 0.5 * s.cumulative_shift

        for i in (IYZ, IXZ, IXY):
            s = sets[i]
            if s.style != "pressure":
                continue
            # yz and xz shear across z, xy across y.
            length = float(prd[1 if i == IXY else 2])
            rate = limit(s.pgain * (p_current[i] - s.ptarget), s.ptarget)
            s.cumulative_shift += dt * rate * length
            s.tilt_target = s.tilt_start + s.cumulative_shift

    def _adjust_linked_rates(
        self, e_larger: float, e_smaller: float, e3: float, vi: float, v: float, dt: float
    ) -> tuple[float, float]:
        """Rescale a volume-preserving pair of strain rates to honour max/rate."""
        m = self.max_rate
        e_lim_pos = (vi - v * (1 + m * dt)) / (v * (1 + m * dt) * dt)
        e_lim_neg = (vi - v * (1 - m * dt)) / (v * (1 - m * dt) * dt)
        if e_larger * e3 >= 0:
            if e_larger > 0.0:
                return e_lim_neg, -m
            return e_lim_pos, m
        if e_larger > 0.0:
            return m, e_lim_pos
        return -m, e_lim_neg

    def _apply_volume(self, dt: float) -> None:
        """Derive the size of `volume` dimensions from the ones they follow."""
        assert self.domain is not None
        linked_done = False
        e2_linked = 0.0

        for i in range(3):
            s = self.sets[i]
            if s.style != "volume":
                continue
            v0 = s.vol_start
            if s.substyle == "one_from_one":
                d1, fx = self.sets[s.dynamic1], self.sets[s.fixed]
                half = 0.5 * (
                    v0 / (d1.hi_target - d1.lo_target) / (fx.hi_start - fx.lo_start)
                )
            elif s.substyle == "one_from_two":
                d1, d2 = self.sets[s.dynamic1], self.sets[s.dynamic2]
                half = 0.5 * (
                    v0
                    / (d1.hi_target - d1.lo_target)
                    / (d2.hi_target - d2.lo_target)
                )
            elif not self.vol_balance_p:
                d1, fx = self.sets[s.dynamic1], self.sets[s.fixed]
                half = 0.5 * math.sqrt(
                    v0
                    * (s.hi_start - s.lo_start)
                    / (d1.hi_target - d1.lo_target)
                    / (fx.hi_start - fx.lo_start)
                )
            else:
                half, linked_done, e2_linked = self._balanced_volume_half(
                    i, s, dt, linked_done, e2_linked
                )
            mid = 0.5 * (s.lo_start + s.hi_start)
            s.lo_target, s.hi_target = mid - half, mid + half

    def _balanced_volume_half(
        self, i: int, s: AxisSet, dt: float, linked_done: bool, e2_linked: float
    ) -> tuple[float, bool, float]:
        """
        The vol/balance/p branch of FixDeformPressure::apply_volume().

        Two dimensions hold the volume constant while their strain rates are
        split so that their pressures converge, by expanding the stress to
        linear order in the rates measured on the previous step.
        """
        assert self.domain is not None
        fx = self.sets[s.fixed]
        d1 = self.sets[s.dynamic1]

        l1i = float(self.domain.prd[i])
        l2i = float(self.domain.prd[s.fixed])
        l3i = float(self.domain.prd[s.dynamic1])
        l3 = d1.hi_target - d1.lo_target
        vi = l1i * l2i * l3i
        v = l3 * l1i * l2i
        e3 = (l3 / l3i - 1.0) / dt

        e1i, e2i = s.prior_rate, fx.prior_rate
        p1, p2 = s.pressure_now, fx.pressure_now  # type: ignore[attr-defined]
        p1i, p2i = s.prior_pressure, fx.prior_pressure

        if e3 == 0.0:
            return 0.5 * l1i, linked_done, e2_linked
        if e1i == 0.0 or e2i == 0.0 or (p2 == p2i and p1 == p1i):
            # No prior strain, or no pressure response yet: fall back to the
            # plain volume-preserving split.
            return (
                0.5 * math.sqrt(s.vol_start * l1i / l3 / l2i),
                linked_done,
                e2_linked,
            )
        if linked_done:
            return 0.5 * l1i * (1.0 + e2_linked * dt), linked_done, e2_linked

        denominator = p2 - p2i + e2i * ((p1 - p1i) / e1i)
        if denominator != 0.0:
            e1 = (((p2 - p2i) * (vi - v) / (v * dt)) - e2i * (p1 - p2)) / denominator
        else:
            e1 = e2i
        e2 = (vi - v * (1 + e1 * dt)) / (v * (1 + e1 * dt) * dt)

        if self.max_rate != 0.0 and (
            abs(e1) > self.max_rate or abs(e2) > self.max_rate
        ):
            if abs(e1) > abs(e2):
                e1, e2 = self._adjust_linked_rates(e1, e2, e3, vi, v, dt)
            else:
                e2, e1 = self._adjust_linked_rates(e2, e1, e3, vi, v, dt)

        return 0.5 * l1i * (1.0 + e1 * dt), True, e2

    def _apply_box(self, tensor: np.ndarray, dt: float) -> None:
        """Final isotropic scaling of the whole box (FixDeformPressure::apply_box)."""
        assert self.domain is not None
        sb = self.set_box
        if sb.style == "none":
            return

        if sb.style == "volume":
            v = 1.0
            for i in range(3):
                v *= self.sets[i].hi_target - self.sets[i].lo_target
            scale = (sb.vol_start / v) ** (1.0 / 3.0)
            for i in range(3):
                s = self.sets[i]
                half = 0.5 * (s.hi_target - s.lo_target) * scale
                mid = 0.5 * (s.lo_start + s.hi_start)
                s.lo_target, s.hi_target = mid - half, mid + half
            return

        # box pressure: proportional control on the scalar mean pressure
        v_rate = self._limit_rate(
            sb.pgain * (float(np.mean(tensor[:3])) - sb.ptarget), sb.ptarget
        )
        for i in range(3):
            s = self.sets[i]
            shift = (s.hi_target - s.lo_target) * dt * v_rate
            sb.cumulative_vshift[i] += shift
            if s.style == "none":
                # Overwrite the default target of the current length.
                s.lo_target, s.hi_target = s.lo_start, s.hi_start
            s.lo_target -= 0.5 * sb.cumulative_vshift[i]
            s.hi_target += 0.5 * sb.cumulative_vshift[i]

    # ------------------------------------------------------------------ domain

    def _normalize_and_flip(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Bring tilt targets next to the current tilt, and flip a skewed box.

        LAMMPS FixDeform::update_domain(): a tilt target may be many box lengths
        away (a `variable` or `final` target), so it is first reduced to the
        equivalent value closest to the current one.  Then, if any tilt exceeds
        half its box length, the lattice vectors are relabelled.

        Two tilts come back. The *deformed* one is the normalised target, and is
        what the atoms are remapped onto -- a flip must not move atoms, it only
        renames the cell edges, and LAMMPS likewise remaps with the un-flipped
        target and applies the flip afterwards. The *final* one carries the flip
        and is what the box is set to.

        Unlike LAMMPS, which tracks image flags and migrates atoms, all
        taichimps has to do after a flip is ask for a rebuild: the wrap that
        happens there puts everything back inside the relabelled cell.
        """
        assert self.domain is not None
        dom = self.domain
        sets = self.sets
        new_lo = np.array([sets[i].lo_target for i in range(3)])
        new_hi = np.array([sets[i].hi_target for i in range(3)])
        # Domain order [xy, xz, yz]
        tilt = np.array(
            [sets[IXY].tilt_target, sets[IXZ].tilt_target, sets[IYZ].tilt_target]
        )
        cur = dom.tilt
        dom_prd = dom.prd
        prd = new_hi - new_lo

        # Reduce each target to the equivalent nearest the current value. xy and
        # xz are measured against x, yz against y.
        for set_i, dom_i in ((IXY, 0), (IYZ, 2), (IXZ, 1)):
            denom = prd[1] if set_i == IYZ else prd[0]
            if denom <= 0.0:
                continue
            current = cur[dom_i] / dom_prd[1 if set_i == IYZ else 0]
            n = round(tilt[dom_i] / denom - current)
            if n != 0:
                tilt[dom_i] -= n * denom
                if set_i == IYZ:
                    # Shifting c by a multiple of b also shifts its x component.
                    tilt[1] -= n * tilt[0]

        deformed = tilt.copy()
        flipped = False
        if self.flip:
            over = (
                abs(tilt[2]) > 0.5 * prd[1]      # yz vs yprd
                or abs(tilt[1]) > 0.5 * prd[0]   # xz vs xprd
                or abs(tilt[0]) > 0.5 * prd[0]   # xy vs xprd
            )
            if over:
                if dom.pbc_y:
                    if tilt[2] < -0.5 * prd[1]:
                        tilt[2] += prd[1]
                        tilt[1] += tilt[0]
                        flipped = True
                    elif tilt[2] > 0.5 * prd[1]:
                        tilt[2] -= prd[1]
                        tilt[1] -= tilt[0]
                        flipped = True
                if dom.pbc_x:
                    if tilt[1] < -0.5 * prd[0]:
                        tilt[1] += prd[0]
                        flipped = True
                    elif tilt[1] > 0.5 * prd[0]:
                        tilt[1] -= prd[0]
                        flipped = True
                    if tilt[0] < -0.5 * prd[0]:
                        tilt[0] += prd[0]
                        flipped = True
                    elif tilt[0] > 0.5 * prd[0]:
                        tilt[0] -= prd[0]
                        flipped = True

        return new_lo, new_hi, deformed, tilt, flipped

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        if self.domain is None:
            return
        if not self._started:
            # Tolerate callers that never run setup() explicitly; latch on the
            # first step instead, which is what a bare `run` would do anyway.
            self.setup(0)

        self.step_count += 1
        self.nsteps += 1
        if self.step_count % self.nevery != 0:
            return

        # Bound to locals for the same reason as in _apply_strain: attribute
        # access on a @ti.data_oriented object is not free, and this is the
        # per-step path.
        dom = self.domain
        sets = self.sets

        needs_pressure = (
            any(s.style in PRESSURE_STYLES for s in sets)
            or self.set_box.style == "pressure"
            or self.vol_balance_p
        )
        tensor = np.zeros(6)
        p_current = np.zeros(6)
        if needs_pressure:
            p_current, tensor = self._current_pressure(atom)
            for i in range(3):
                s = sets[i]
                s.pressure_now = float(tensor[i])  # type: ignore[attr-defined]
                if not s.saved:
                    s.saved = True
                    s.prior_rate = 0.0
                    s.prior_pressure = float(tensor[i])

        old_prd = dom.prd.copy()
        old_lo = dom.boxlo.copy()
        old_tilt = dom.tilt.copy()

        self._apply_strain(dt)
        if needs_pressure:
            self._apply_pressure(p_current, dt)
        self._apply_volume(dt)
        if self.set_box.style != "none":
            self._apply_box(tensor, dt)

        new_lo, new_hi, deformed_tilt, new_tilt, flipped = self._normalize_and_flip()
        new_prd = new_hi - new_lo

        if np.any(new_prd <= 0.0):
            raise RuntimeError(
                f"fix deform/pressure drove a box dimension to {new_prd}, which is "
                "not positive; check the gain, max/rate or strain rate"
            )

        if needs_pressure:
            # Remember this step's response for the next vol/balance/p update.
            for i in range(3):
                s = sets[i]
                s.prior_pressure = float(tensor[i])
                s.prior_rate = (
                    (s.hi_target - s.lo_target) / old_prd[i] - 1.0
                ) / dt

        # LAMMPS keeps domain->h_rate as box length (and tilt) rates, not strain
        # rates; `remap v` reads it when wrapping an atom across a boundary.
        # For every style whose target is linear in time this finite difference
        # is exactly the rate LAMMPS stores analytically; `trate` differs only
        # at O(rate*dt).
        h_rate = np.zeros(6)
        h_rate[:3] = (new_prd - old_prd) / dt
        # h_rate order past the diagonal is [yz, xz, xy]; new_tilt is [xy, xz, yz]
        h_rate[3] = (new_tilt[2] - old_tilt[2]) / dt
        h_rate[4] = (new_tilt[1] - old_tilt[1]) / dt
        h_rate[5] = (new_tilt[0] - old_tilt[0]) / dt

        # The box itself is published below, after the `remap x` pass: that
        # pass reads the box out of Domain and needs to see the old one.
        if self.remap == "x" and atom.nlocal > 0:
            xprd, yprd, zprd = old_prd
            oxy, oxz, oyz = old_tilt
            self._old_lo[None] = ti.Vector(old_lo.tolist())
            self._new_lo[None] = ti.Vector(new_lo.tolist())
            self._old_prd[None] = ti.Vector(old_prd.tolist())
            self._old_hinv_tilt[None] = ti.Vector([
                -oxy / (xprd * yprd),
                (oyz * oxy - yprd * oxz) / (xprd * yprd * zprd),
                -oyz / (yprd * zprd),
            ])
            self._new_prd[None] = ti.Vector(new_prd.tolist())
            # Deform the atoms onto the un-flipped cell; the flip below is a
            # relabelling that must leave them where they are.
            self._new_tilt[None] = ti.Vector(deformed_tilt.tolist())
            self.remap_positions(atom.nlocal, atom.x)

        dom.set_box_and_h_rate(
            new_lo.tolist(), new_hi.tolist(), new_tilt.tolist(),
            h_rate, vremap=(self.remap == "v"),
        )
        if flipped:
            # Atoms now sit outside the relabelled cell until they are wrapped.
            self.force_reneighbor = True
