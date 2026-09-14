"""
Simulation domain and periodic boundary conditions.
Reference: LAMMPS src/domain.cpp
"""

from typing import Any

import numpy as np
import taichi as ti


@ti.data_oriented
class Domain:
    """
    Simulation domain [boxlo, boxhi] with periodic or fixed boundaries, and
    optional tilt factors for a restricted triclinic box.

    bcond: 'p' (periodic), 'f' (fixed/non-periodic).
    e.g. boundary=['p', 'p', 'f'] for x, y periodic, z fixed.

    Shape matrix, in the LAMMPS ordering h = [xprd, yprd, zprd, yz, xz, xy]:

        a = (xprd,  0,    0   )
        b = (xy,    yprd, 0   )
        c = (xz,    yz,   zprd)

    so h and its inverse are upper triangular.  Lamda ("fractional") coordinates
    are lamda = h_inv . (x - boxlo), in which the box is always the unit cube.

    The box is stored twice on purpose.  `boxlo`/`boxhi`/`prd`/`periodicity`/
    `tilt` are plain numpy arrays for host code to read and index, while
    `boxlo_f` and friends are zero-dimensional Taichi fields that the kernels
    read.

    That split is not cosmetic.  These used to be `ti.Vector` objects, which are
    Python values: Taichi bakes them into a kernel as compile-time constants the
    first time it is compiled, and later assignments never reach the compiled
    code.  Any run with a changing box -- every `fix deform/pressure` run, i.e.
    every isotropic-compression or triaxial test -- therefore kept evaluating
    minimum-image distances, PBC wrapping and neighbor cell indices against the
    box as it was at step zero.  Reading from a field makes the kernels see the
    current box.
    """

    def __init__(
        self,
        boxlo: list[float] | tuple[float, float, float],
        boxhi: list[float] | tuple[float, float, float],
        boundary: list[str] | tuple[str, str, str] = ("p", "p", "p"),
        tilt: list[float] | tuple[float, float, float] | None = None,
        float_type: Any = ti.f64,
    ) -> None:
        self.float_type = float_type

        # Device-side copies, read by every kernel.
        self.boxlo_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.boxhi_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.prd_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.periodicity_f = ti.Vector.field(3, dtype=ti.i32, shape=())
        # [xy, xz, yz] and the inverse-shape off-diagonals [h_inv5, h_inv4, h_inv3]
        self.tilt_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.hinv_tilt_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.triclinic_f = ti.field(dtype=ti.i32, shape=())

        # Host-side mirrors, kept in step by set_box()/set_tilt()/set_boundary().
        self.boxlo = np.array([float(v) for v in boxlo], dtype=np.float64)
        self.boxhi = np.array([float(v) for v in boxhi], dtype=np.float64)
        self.prd = self.boxhi - self.boxlo
        self.periodicity = np.array(
            [1 if b == "p" else 0 for b in boundary], dtype=np.int32
        )
        # [xy, xz, yz], matching the LAMMPS data-file ordering.
        self.tilt = np.zeros(3, dtype=np.float64)
        if tilt is not None:
            self.tilt = np.array([float(v) for v in tilt], dtype=np.float64)
        self._sync()

    # ------------------------------------------------------------------ state

    @property
    def triclinic(self) -> bool:
        """True when any tilt factor is non-zero."""
        return bool(np.any(self.tilt != 0.0))

    @property
    def xy(self) -> float:
        return float(self.tilt[0])

    @property
    def xz(self) -> float:
        return float(self.tilt[1])

    @property
    def yz(self) -> float:
        return float(self.tilt[2])

    def _sync(self) -> None:
        """Push the host-side box onto the device."""
        xy, xz, yz = (float(v) for v in self.tilt)
        xprd, yprd, zprd = (float(v) for v in self.prd)

        # Domain::set_global_box(): h = [xprd, yprd, zprd, yz, xz, xy]
        #   h_inv[3] = -yz / (yprd*zprd)
        #   h_inv[4] = (yz*xy - yprd*xz) / (xprd*yprd*zprd)
        #   h_inv[5] = -xy / (xprd*yprd)
        hinv_xy = -xy / (xprd * yprd)
        hinv_xz = (yz * xy - yprd * xz) / (xprd * yprd * zprd)
        hinv_yz = -yz / (yprd * zprd)

        self.boxlo_f[None] = ti.Vector(self.boxlo.tolist())
        self.boxhi_f[None] = ti.Vector(self.boxhi.tolist())
        self.prd_f[None] = ti.Vector(self.prd.tolist())
        self.periodicity_f[None] = ti.Vector(self.periodicity.tolist())
        self.tilt_f[None] = ti.Vector([xy, xz, yz])
        self.hinv_tilt_f[None] = ti.Vector([hinv_xy, hinv_xz, hinv_yz])
        self.triclinic_f[None] = 1 if self.triclinic else 0

    @property
    def volume(self) -> float:
        """
        Volume of the domain box.

        det(h) = xprd * yprd * zprd for a restricted triclinic box too, since h
        is triangular: tilting shears the cell without changing its volume.
        """
        return float(self.prd[0] * self.prd[1] * self.prd[2])

    def set_box(
        self,
        boxlo: list[float] | tuple[float, float, float],
        boxhi: list[float] | tuple[float, float, float],
        tilt: list[float] | tuple[float, float, float] | None = None,
    ) -> None:
        """Update domain bounds (and optionally tilt), on the host and device."""
        self.boxlo = np.array([float(v) for v in boxlo], dtype=np.float64)
        self.boxhi = np.array([float(v) for v in boxhi], dtype=np.float64)
        self.prd = self.boxhi - self.boxlo
        if tilt is not None:
            self.tilt = np.array([float(v) for v in tilt], dtype=np.float64)
        self._sync()

    def set_tilt(self, tilt: list[float] | tuple[float, float, float]) -> None:
        """Update the [xy, xz, yz] tilt factors."""
        self.tilt = np.array([float(v) for v in tilt], dtype=np.float64)
        self._sync()

    def set_boundary(self, boundary: list[str] | tuple[str, str, str]) -> None:
        """Update boundary periodicity."""
        self.periodicity = np.array(
            [1 if b == "p" else 0 for b in boundary], dtype=np.int32
        )
        self._sync()

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Axis-aligned bounding box of the (possibly tilted) cell.

        This is LAMMPS's boxlo_bound/boxhi_bound, which is what a dump file
        reports for a triclinic box.
        """
        xy, xz, yz = (float(v) for v in self.tilt)
        lo = self.boxlo.copy()
        hi = self.boxhi.copy()
        lo[0] = min(lo[0], lo[0] + xy, lo[0] + xz, lo[0] + xy + xz)
        hi[0] = max(hi[0], hi[0] + xy, hi[0] + xz, hi[0] + xy + xz)
        lo[1] = min(lo[1], lo[1] + yz)
        hi[1] = max(hi[1], hi[1] + yz)
        return lo, hi

    @property
    def pbc_x(self) -> bool:
        return bool(self.periodicity[0])

    @property
    def pbc_y(self) -> bool:
        return bool(self.periodicity[1])

    @property
    def pbc_z(self) -> bool:
        return bool(self.periodicity[2])

    # ----------------------------------------------------------- device funcs

    @ti.func
    def to_lamda(self, delta):
        """
        Map a Cartesian displacement to lamda (fractional) units: h_inv . delta.

        Only the displacement form is needed here; add boxlo yourself for an
        absolute coordinate. h_inv is upper triangular, so this subtracts the
        tilt contributions from each component in turn.
        """
        prd = self.prd_f[None]
        ht = self.hinv_tilt_f[None]
        return ti.Vector([
            delta[0] / prd[0] + ht[0] * delta[1] + ht[1] * delta[2],
            delta[1] / prd[1] + ht[2] * delta[2],
            delta[2] / prd[2],
        ])

    @ti.func
    def to_cartesian(self, lamda):
        """Map a lamda displacement back to Cartesian: h . lamda."""
        prd = self.prd_f[None]
        t = self.tilt_f[None]
        return ti.Vector([
            prd[0] * lamda[0] + t[0] * lamda[1] + t[1] * lamda[2],
            prd[1] * lamda[1] + t[2] * lamda[2],
            prd[2] * lamda[2],
        ])

    @ti.func
    def minimum_image(self, dx):
        """
        Apply the minimum image convention.

        Closed form rather than a `while` loop: the loop form spins forever if a
        coordinate ever becomes non-finite, turning a diverging simulation into
        a hang instead of a visible NaN.

        A tilted box is reduced one lattice vector at a time, in LAMMPS's order:
        z first, correcting y and x by yz and xz, then y correcting x by xy,
        then x.  The order matters and the corrections must use the *rounded*
        image count, so this cannot be collapsed into rounding all three lamda
        components at once -- doing that corrects y using the unrounded z
        fraction and lands a full box length away whenever that shifts the y
        component across a rounding boundary.
        """
        prd = self.prd_f[None]
        per = self.periodicity_f[None]
        if self.triclinic_f[None] == 0:
            for dim in ti.static(range(3)):
                if per[dim] == 1:
                    dx[dim] -= prd[dim] * ti.round(dx[dim] / prd[dim])
        else:
            t = self.tilt_f[None]  # [xy, xz, yz]
            if per[2] == 1:
                nz = ti.round(dx[2] / prd[2])
                dx[2] -= nz * prd[2]
                dx[1] -= nz * t[2]
                dx[0] -= nz * t[1]
            if per[1] == 1:
                ny = ti.round(dx[1] / prd[1])
                dx[1] -= ny * prd[1]
                dx[0] -= ny * t[0]
            if per[0] == 1:
                dx[0] -= ti.round(dx[0] / prd[0]) * prd[0]
        return dx

    @ti.func
    def pbc_wrap(self, x):
        """
        Wrap coordinates into the box, as LAMMPS Domain::pbc() does.

        LAMMPS wraps a triclinic box in lamda coordinates (x2lamda, pbc,
        lamda2x); the same thing happens here, in one step.
        """
        lo = self.boxlo_f[None]
        prd = self.prd_f[None]
        per = self.periodicity_f[None]
        if self.triclinic_f[None] == 0:
            for dim in ti.static(range(3)):
                if per[dim] == 1:
                    s = (x[dim] - lo[dim]) / prd[dim]
                    x[dim] = lo[dim] + (s - ti.floor(s)) * prd[dim]
        else:
            lamda = self.to_lamda(x - lo)
            for dim in ti.static(range(3)):
                if per[dim] == 1:
                    lamda[dim] -= ti.floor(lamda[dim])
            x = self.to_cartesian(lamda) + lo
        return x

    @ti.func
    def lamda_of(self, x):
        """Absolute position in lamda coordinates: h_inv . (x - boxlo)."""
        return self.to_lamda(x - self.boxlo_f[None])

    @ti.kernel
    def _pbc_kernel(self, nlocal: ti.i32, x: ti.template()):
        for i in range(nlocal):
            x[i] = self.pbc_wrap(x[i])

    def pbc(self, atom: Any) -> None:
        """Apply periodic boundary conditions to all atom positions."""
        self._pbc_kernel(atom.nlocal, atom.x)
