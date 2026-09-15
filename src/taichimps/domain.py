"""
Simulation domain and periodic boundary conditions.
Reference: LAMMPS src/domain.cpp
"""

from typing import Any

import numpy as np
import taichi as ti

# Layout of the Domain._sync staging buffer:
#   0-2 boxlo, 3-5 boxhi, 6-8 tilt, 9-11 periodicity,
#   12-17 h_rate, 18 vremap, 19 triclinic
_STAGE_LEN = 20


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
        # Rate of change of the box, in LAMMPS's h_rate ordering
        # [xprd, yprd, zprd, yz, xz, xy] -- box *length* rates, not strain
        # rates. Only used when a deforming fix asks for `remap v`.
        self.h_rate_f = ti.Vector.field(6, dtype=float_type, shape=())
        self.vremap_f = ti.field(dtype=ti.i32, shape=())

        # Staging buffer for _sync(). Writing the nine fields above one at a
        # time costs nine host-to-device round trips; a deforming run does that
        # twice per step (set_box and set_h_rate), which measured at 0.64 ms per
        # step on CUDA -- more than three times the cost of the contact force
        # kernel itself. One transfer into this buffer plus one kernel that
        # unpacks it on the device does the same job for a quarter of that.
        self._stage = ti.field(dtype=float_type, shape=_STAGE_LEN)
        self._stage_host = np.zeros(_STAGE_LEN, dtype=np.float64)

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
        # [xprd, yprd, zprd, yz, xz, xy] rates; set by a deforming fix.
        self.h_rate = np.zeros(6, dtype=np.float64)
        self.vremap = False
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

    @ti.kernel
    def _unpack_stage(self):
        """
        Spread the staging buffer over the fields the kernels actually read.

        prd and the off-diagonals of h_inv are derived here rather than sent,
        so the host only ships the box itself.
        """
        st = ti.static(self._stage)
        lo = ti.Vector([st[0], st[1], st[2]])
        hi = ti.Vector([st[3], st[4], st[5]])
        xy, xz, yz = st[6], st[7], st[8]

        self.boxlo_f[None] = lo
        self.boxhi_f[None] = hi
        prd = hi - lo
        self.prd_f[None] = prd
        self.tilt_f[None] = ti.Vector([xy, xz, yz])
        self.periodicity_f[None] = ti.Vector([
            ti.cast(st[9], ti.i32), ti.cast(st[10], ti.i32), ti.cast(st[11], ti.i32)
        ])
        self.h_rate_f[None] = ti.Vector([st[12], st[13], st[14],
                                         st[15], st[16], st[17]])
        self.vremap_f[None] = ti.cast(st[18], ti.i32)
        self.triclinic_f[None] = ti.cast(st[19], ti.i32)

        # Domain::set_global_box(): h = [xprd, yprd, zprd, yz, xz, xy]
        #   h_inv[3] = -yz / (yprd*zprd)
        #   h_inv[4] = (yz*xy - yprd*xz) / (xprd*yprd*zprd)
        #   h_inv[5] = -xy / (xprd*yprd)
        xprd, yprd, zprd = prd[0], prd[1], prd[2]
        self.hinv_tilt_f[None] = ti.Vector([
            -xy / (xprd * yprd),
            (yz * xy - yprd * xz) / (xprd * yprd * zprd),
            -yz / (yprd * zprd),
        ])

    def _sync(self) -> None:
        """Push the host-side box onto the device, in a single transfer."""
        st = self._stage_host
        st[0:3] = self.boxlo
        st[3:6] = self.boxhi
        st[6:9] = self.tilt
        st[9:12] = self.periodicity
        st[12:18] = self.h_rate
        st[18] = 1.0 if self.vremap else 0.0
        st[19] = 1.0 if self.triclinic else 0.0
        self._stage.from_numpy(st)
        self._unpack_stage()

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

    def set_box_and_h_rate(
        self,
        boxlo: list[float] | tuple[float, float, float],
        boxhi: list[float] | tuple[float, float, float],
        tilt: list[float] | tuple[float, float, float] | None,
        h_rate: Any,
        vremap: bool,
    ) -> None:
        """
        Set the box and its rate of change together, in one device transfer.

        A deforming fix updates both every step; doing it as set_box() followed
        by set_h_rate() syncs the device twice for no reason.
        """
        self.boxlo = np.array([float(v) for v in boxlo], dtype=np.float64)
        self.boxhi = np.array([float(v) for v in boxhi], dtype=np.float64)
        self.prd = self.boxhi - self.boxlo
        if tilt is not None:
            self.tilt = np.array([float(v) for v in tilt], dtype=np.float64)
        self.h_rate = np.asarray(h_rate, dtype=np.float64)
        self.vremap = bool(vremap)
        self._sync()

    def set_h_rate(self, h_rate: Any, vremap: bool) -> None:
        """
        Record how fast the box is changing, for `remap v`.

        LAMMPS keeps this in domain->h_rate and Domain::pbc() uses it to adjust
        the velocity of an atom that crosses a periodic boundary, so the atom
        arrives on the far side with the velocity the deforming lattice implies.
        The units are a box length (or tilt) per unit time, not a strain rate --
        that distinction is the subject of a LAMMPS bug this project reported.
        """
        self.h_rate = np.asarray(h_rate, dtype=np.float64)
        self.vremap = bool(vremap)
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
    def minimum_image_and_vshift(self, dx):
        """
        Minimum image of a separation, plus the partner's velocity shift.

        Under `remap v` a periodic image of an atom moves with the deforming
        lattice, so a contact that reaches across a boundary sees a partner
        whose velocity is offset by the box rate.  LAMMPS gets this for free:
        AtomVec::pack_comm_vel() adds `pbc . h_rate` to a ghost's velocity when
        deform_vremap is set.  taichimps has no ghosts, so the shift has to be
        applied here, to the same image count the minimum image convention just
        used.  Without it the damping force on every boundary-spanning contact
        is wrong, which is exactly the coupling the upstream h_rate bug report
        describes.

        Returns (reduced separation, velocity to ADD to the partner).
        """
        n = self.image_offsets(dx)
        dv = ti.Vector([0.0, 0.0, 0.0])
        if self.vremap_f[None] != 0:
            h = self.h_rate_f[None]
            dv = ti.Vector([
                n[0] * h[0] + n[1] * h[5] + n[2] * h[4],
                n[1] * h[1] + n[2] * h[3],
                n[2] * h[2],
            ])
        return self.apply_image_offsets(dx, n), dv

    @ti.func
    def image_offsets(self, dx):
        """
        Image counts the minimum image convention removes from a separation.

        Reduced one lattice vector at a time in LAMMPS's order -- z first
        (correcting y and x by yz and xz), then y correcting x by xy, then x --
        because each correction has to use the *rounded* image count of the
        previous direction.
        """
        prd = self.prd_f[None]
        per = self.periodicity_f[None]
        n = ti.Vector([0.0, 0.0, 0.0])
        if self.triclinic_f[None] == 0:
            for dim in ti.static(range(3)):
                if per[dim] == 1:
                    n[dim] = ti.round(dx[dim] / prd[dim])
        else:
            t = self.tilt_f[None]  # [xy, xz, yz]
            d = dx
            if per[2] == 1:
                n[2] = ti.round(d[2] / prd[2])
                d[2] -= n[2] * prd[2]
                d[1] -= n[2] * t[2]
                d[0] -= n[2] * t[1]
            if per[1] == 1:
                n[1] = ti.round(d[1] / prd[1])
                d[1] -= n[1] * prd[1]
                d[0] -= n[1] * t[0]
            if per[0] == 1:
                n[0] = ti.round(d[0] / prd[0])
        return n

    @ti.func
    def apply_image_offsets(self, dx, n):
        """Subtract `n` lattice vectors from a separation."""
        prd = self.prd_f[None]
        t = self.tilt_f[None]  # [xy, xz, yz]
        out = dx
        if self.triclinic_f[None] == 0:
            for dim in ti.static(range(3)):
                out[dim] -= n[dim] * prd[dim]
        else:
            out[2] -= n[2] * prd[2]
            out[1] -= n[2] * t[2] + n[1] * prd[1]
            out[0] -= n[2] * t[1] + n[1] * t[0] + n[0] * prd[0]
        return out

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
    def image_count(self, x):
        """How many box lengths, per lattice direction, the point is out by."""
        lamda = self.to_lamda(x - self.boxlo_f[None])
        per = self.periodicity_f[None]
        n = ti.Vector([0.0, 0.0, 0.0])
        for dim in ti.static(range(3)):
            if per[dim] == 1:
                n[dim] = ti.floor(lamda[dim])
        return n

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
    def vremap_delta(self, n):
        """
        Velocity correction for an atom that moved `n` box lengths, for remap v.

        Domain::pbc() applies, per crossing: leaving through x adjusts vx by
        h_rate[0]; through y, vx by h_rate[5] (the xy tilt rate) and vy by
        h_rate[1]; through z, vx by h_rate[4], vy by h_rate[3] and vz by
        h_rate[2]. Signs follow the direction of the crossing, which `n` already
        carries, and a multi-period jump scales with it.
        """
        h = self.h_rate_f[None]
        return -ti.Vector([
            n[0] * h[0] + n[1] * h[5] + n[2] * h[4],
            n[1] * h[1] + n[2] * h[3],
            n[2] * h[2],
        ])

    @ti.func
    def lamda_of(self, x):
        """Absolute position in lamda coordinates: h_inv . (x - boxlo)."""
        return self.to_lamda(x - self.boxlo_f[None])

    @ti.kernel
    def _pbc_kernel(self, nlocal: ti.i32, x: ti.template()):
        for i in range(nlocal):
            x[i] = self.pbc_wrap(x[i])

    @ti.kernel
    def _pbc_vremap_kernel(self, nlocal: ti.i32, x: ti.template(), v: ti.template()):
        for i in range(nlocal):
            n = self.image_count(x[i])
            if n[0] != 0.0 or n[1] != 0.0 or n[2] != 0.0:
                v[i] += self.vremap_delta(n)
            x[i] = self.pbc_wrap(x[i])

    def pbc(self, atom: Any) -> None:
        """
        Apply periodic boundary conditions to all atom positions.

        Under `remap v` the velocity of a crossing atom is corrected by the box
        deformation rate, which is the only way the deformation reaches the
        atoms in that mode -- their coordinates are not remapped affinely.
        """
        if self.vremap:
            self._pbc_vremap_kernel(atom.nlocal, atom.x, atom.v)
        else:
            self._pbc_kernel(atom.nlocal, atom.x)
