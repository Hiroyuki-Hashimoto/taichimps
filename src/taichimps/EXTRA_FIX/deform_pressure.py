"""
Box deformation with optional pressure servo control.

Reference: LAMMPS src/EXTRA-FIX/fix_deform_pressure.cpp and src/fix_deform.cpp
License: GPL v2 compatible / MIT reimplementation

Each of x, y, z carries its own control style, and strain-controlled and
pressure-controlled axes can be mixed -- which is what makes the standard
triaxial setup expressible in a single fix:

    fix 1 all deform/pressure 1 z trate -0.01 \\
        x pressure 100000.0 1e-4 y pressure 100000.0 1e-4 max/rate 0.1

Implemented styles
    x/y/z    none | erate | trate | vel | final | scale | delta | volume |
             pressure | pressure/mean
    keywords couple, max/rate, normalize/pressure, remap, nevery

Not implemented: the tilt factors xy/xz/yz (Domain is orthogonal only, so
there is nothing to shear), `wiggle`, `variable` targets, `box` scaling and
`vol/balance/p`.  Those raise rather than being silently ignored.

Box updates follow the LAMMPS bookkeeping: targets are always expressed
relative to the box as it was at the start of the run (`lo_start`/`hi_start`)
plus a `cumulative_shift`, rather than as an increment on the current box, so
the trajectory does not depend on rounding of intermediate states.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.computes.thermo import Computes
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix

STRAIN_STYLES = ("erate", "trate", "vel", "final", "scale", "delta")
PRESSURE_STYLES = ("pressure", "pressure/mean")
VALID_STYLES = ("none", "volume", *STRAIN_STYLES, *PRESSURE_STYLES)

COUPLE_CHOICES = ("none", "xyz", "xy", "yz", "xz")

# Styles whose target is interpolated between the box at run start and a box at
# run end, so they need to know how long the run is.
NEEDS_RUN_LENGTH = ("final", "scale", "delta")


@dataclass
class AxisSet:
    """Per-dimension control settings, the equivalent of LAMMPS `Set`."""

    style: str = "none"
    # strain parameters
    rate: float = 0.0          # erate / trate
    vel: float = 0.0           # vel
    scale: float = 1.0         # scale
    flo: float = 0.0           # final
    fhi: float = 0.0
    dlo: float = 0.0           # delta
    dhi: float = 0.0
    # pressure parameters
    ptarget: float = 0.0
    pgain: float = 0.0
    # run-time state
    lo_start: float = 0.0
    hi_start: float = 0.0
    lo_target: float = 0.0
    hi_target: float = 0.0
    cumulative_shift: float = 0.0
    vol_start: float = 0.0
    # volume coupling
    substyle: str = ""
    fixed: int = -1
    dynamic1: int = -1
    dynamic2: int = -1
    coupled: bool = False


@ti.data_oriented
class FixDeformPressure(Fix):
    """
    Deform the periodic box, optionally servo-controlling it to a target pressure.

    A pressure-controlled axis uses proportional control on the engineering
    strain rate, as in FixDeformPressure::apply_pressure():

        strain_rate = pgain * (p_current - p_target)

    `normalize_pressure` divides that by |p_target| so the gain becomes
    dimensionless, and `max_rate` caps the magnitude.  `pressure` follows the
    matching diagonal component of the pressure tensor, `pressure/mean` follows
    the scalar mean pressure.
    """

    def __init__(
        self,
        domain: Domain,
        axes: dict[str, dict[str, Any]] | None = None,
        couple: str = "none",
        max_rate: float = 0.0,
        normalize_pressure: bool = False,
        remap: str = "x",
        nevery: int = 1,
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
        if remap not in ("x", "none"):
            raise ValueError("remap must be 'x' or 'none' ('v' is not implemented)")
        self.couple = couple
        self.remap = remap
        self.max_rate = float(max_rate)
        self.normalize_pressure = bool(normalize_pressure)

        self.sets = [AxisSet(), AxisSet(), AxisSet()]
        if axes is None:
            # Isotropic pressure control on all three axes.
            target = 50000.0 if p_target is None else float(p_target)
            gain = 1e-4 if p_gain is None else float(p_gain)
            for s in self.sets:
                s.style = "pressure"
                s.ptarget = target
                s.pgain = gain
            if couple == "none":
                self.couple = "xyz"
        else:
            self._configure_axes(axes)

        self._validate()

        # Scratch fields for the remap kernel. Held as fields rather than
        # passed as scalars so the kernel is compiled once regardless of
        # float_type and of how often the values change.
        self._old_lo = ti.Vector.field(3, dtype=float_type, shape=())
        self._new_lo = ti.Vector.field(3, dtype=float_type, shape=())
        self._scale = ti.Vector.field(3, dtype=float_type, shape=())

        self.computes = Computes(float_type=float_type)
        self.step_count = 0
        self.nsteps = 0
        self.nsteps_total = 0
        self._started = False

    # ------------------------------------------------------------------ setup

    def _configure_axes(self, axes: dict[str, dict[str, Any]]) -> None:
        for name, spec in axes.items():
            if name in ("xy", "xz", "yz"):
                raise ValueError(
                    "tilt control (xy/xz/yz) needs a triclinic Domain, which "
                    "taichimps does not have"
                )
            if name not in ("x", "y", "z"):
                raise ValueError(f"Unknown deform dimension {name!r}")
            idx = "xyz".index(name)
            style = spec["style"]
            if style not in VALID_STYLES:
                raise ValueError(
                    f"Unsupported fix deform/pressure style {style!r}; "
                    f"implemented: {VALID_STYLES}"
                )
            s = self.sets[idx]
            s.style = style
            for key, value in spec.items():
                if key == "style":
                    continue
                if not hasattr(s, key):
                    raise ValueError(f"Unknown parameter {key!r} for dimension {name}")
                setattr(s, key, float(value))

    def _validate(self) -> None:
        for i, s in enumerate(self.sets):
            if s.style in PRESSURE_STYLES and s.pgain <= 0.0:
                raise ValueError(
                    f"fix deform/pressure gain for {'xyz'[i]} must be positive"
                )

        pressure_axes = [s.style in PRESSURE_STYLES for s in self.sets]
        if self.normalize_pressure and not any(pressure_axes):
            raise ValueError(
                "normalize/pressure only applies to pressure-controlled dimensions"
            )
        if self.max_rate < 0.0:
            raise ValueError("max/rate must be positive")

        if self.couple != "none":
            idx = [i for i, c in enumerate("xyz") if c in self.couple]
            ref = None
            for i in idx:
                if self.sets[i].style in PRESSURE_STYLES:
                    ref = i
                    break
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
        for i, s in enumerate(self.sets):
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

    def setup(self, nsteps_total: int = 0) -> None:
        """
        Latch the box at the start of a run, the way FixDeform::init() does.

        `nsteps_total` is only needed by the styles that interpolate towards a
        box at the end of the run (final, scale, delta).
        """
        assert self.domain is not None
        self.nsteps = 0
        self.nsteps_total = int(nsteps_total)
        self._started = True
        for i, s in enumerate(self.sets):
            s.lo_start = float(self.domain.boxlo[i])
            s.hi_start = float(self.domain.boxhi[i])
            s.lo_target = s.lo_start
            s.hi_target = s.hi_start
            s.cumulative_shift = 0.0
            s.vol_start = float(self.domain.volume)
            if s.style in NEEDS_RUN_LENGTH and self.nsteps_total <= 0:
                raise ValueError(
                    f"fix deform/pressure style {s.style!r} needs the run length; "
                    "call setup(nsteps_total) before running"
                )

    # ------------------------------------------------------------------ kernel

    @ti.kernel
    def remap_positions(self, nlocal: ti.i32, x: ti.template()):
        """
        Affine remap, the lamda-coordinate round trip LAMMPS performs.

        x_new = new_lo + (x - old_lo) * (new_prd / old_prd).  Anchoring on the
        *new* lower bound matters: anchoring on the old one (as this used to)
        leaves the particles displaced from the box by half the box-length
        change on every single update.
        """
        old_lo = self._old_lo[None]
        new_lo = self._new_lo[None]
        scale = self._scale[None]
        for i in range(nlocal):
            for d in ti.static(range(3)):
                x[i][d] = new_lo[d] + (x[i][d] - old_lo[d]) * scale[d]

    # ------------------------------------------------------------------- steps

    def _current_pressure(self, atom: AtomSystem) -> np.ndarray:
        """Per-axis pressure to servo on, after applying the couple setting."""
        assert self.domain is not None
        tensor = self.computes.compute_pressure_tensor(atom, self.domain)
        scalar = float(np.mean(tensor[:3]))

        p = np.zeros(3, dtype=np.float64)
        if self.couple == "xyz":
            p[:] = np.mean(tensor[:3])
        elif self.couple in ("xy", "yz", "xz"):
            idx = [i for i, c in enumerate("xyz") if c in self.couple]
            other = [i for i in range(3) if i not in idx]
            p[idx] = np.mean(tensor[idx])
            for i in other:
                p[i] = tensor[i]
        else:
            for i, s in enumerate(self.sets):
                if s.style == "pressure":
                    p[i] = tensor[i]
                elif s.style == "pressure/mean":
                    p[i] = scalar
        return p

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

    def _apply_strain(self, dt: float) -> None:
        """Set targets for the strain-controlled dimensions."""
        assert self.domain is not None
        elapsed = self.nsteps * dt
        frac = 0.0
        if self.nsteps_total > 0:
            frac = self.nsteps / self.nsteps_total

        for i, s in enumerate(self.sets):
            if s.style == "none":
                s.lo_target = float(self.domain.boxlo[i])
                s.hi_target = float(self.domain.boxhi[i])
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

    def _apply_pressure(self, p_current: np.ndarray, dt: float) -> None:
        """Proportional servo on the pressure-controlled dimensions."""
        assert self.domain is not None
        for i, s in enumerate(self.sets):
            if s.style not in PRESSURE_STYLES:
                continue
            rate = s.pgain * (p_current[i] - s.ptarget)
            rate = self._limit_rate(rate, s.ptarget)
            # h_rate is a rate of change of box length, not a strain rate
            shift = dt * rate * float(self.domain.prd[i])
            s.cumulative_shift += shift
            s.lo_target = s.lo_start - 0.5 * s.cumulative_shift
            s.hi_target = s.hi_start + 0.5 * s.cumulative_shift

    def _apply_volume(self) -> None:
        """Derive the size of `volume` dimensions from the ones they follow."""
        for i, s in enumerate(self.sets):
            if s.style != "volume":
                continue
            v0 = s.vol_start
            if s.substyle == "one_from_one":
                d1, fx = self.sets[s.dynamic1], self.sets[s.fixed]
                half = 0.5 * (
                    v0
                    / (d1.hi_target - d1.lo_target)
                    / (fx.hi_start - fx.lo_start)
                )
            elif s.substyle == "one_from_two":
                d1, d2 = self.sets[s.dynamic1], self.sets[s.dynamic2]
                half = 0.5 * (
                    v0
                    / (d1.hi_target - d1.lo_target)
                    / (d2.hi_target - d2.lo_target)
                )
            else:  # two_from_one
                d1, fx = self.sets[s.dynamic1], self.sets[s.fixed]
                half = 0.5 * math.sqrt(
                    v0
                    * (s.hi_start - s.lo_start)
                    / (d1.hi_target - d1.lo_target)
                    / (fx.hi_start - fx.lo_start)
                )
            mid = 0.5 * (s.lo_start + s.hi_start)
            s.lo_target, s.hi_target = mid - half, mid + half

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

        needs_pressure = any(s.style in PRESSURE_STYLES for s in self.sets)
        p_current = (
            self._current_pressure(atom) if needs_pressure else np.zeros(3)
        )

        self._apply_strain(dt)
        if needs_pressure:
            self._apply_pressure(p_current, dt)
        self._apply_volume()

        old_lo = np.array([float(self.domain.boxlo[d]) for d in range(3)])
        old_prd = np.array([float(self.domain.prd[d]) for d in range(3)])
        new_lo = np.array([s.lo_target for s in self.sets])
        new_hi = np.array([s.hi_target for s in self.sets])
        new_prd = new_hi - new_lo

        if np.any(new_prd <= 0.0):
            raise RuntimeError(
                f"fix deform/pressure drove a box dimension to {new_prd}, which is "
                "not positive; check the gain, max/rate or strain rate"
            )

        if self.remap == "x" and atom.nlocal > 0:
            self._old_lo[None] = ti.Vector(old_lo.tolist())
            self._new_lo[None] = ti.Vector(new_lo.tolist())
            self._scale[None] = ti.Vector((new_prd / old_prd).tolist())
            self.remap_positions(atom.nlocal, atom.x)

        self.domain.set_box(new_lo.tolist(), new_hi.tolist())
