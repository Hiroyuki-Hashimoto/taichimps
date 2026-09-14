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
    Orthogonal simulation domain [boxlo, boxhi] with boundary conditions.
    bcond: 'p' (periodic), 'f' (fixed/non-periodic).
    e.g. boundary=['p', 'p', 'f'] for x, y periodic, z fixed.

    The box is stored twice on purpose.  `boxlo`/`boxhi`/`prd`/`periodicity` are
    plain numpy arrays for host code to read and index, while `boxlo_f` and
    friends are zero-dimensional Taichi fields that the kernels read.

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
        float_type: Any = ti.f64,
    ) -> None:
        self.float_type = float_type

        # Device-side copies, read by every kernel.
        self.boxlo_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.boxhi_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.prd_f = ti.Vector.field(3, dtype=float_type, shape=())
        self.periodicity_f = ti.Vector.field(3, dtype=ti.i32, shape=())

        # Host-side mirrors, kept in step by set_box()/set_boundary().
        self.boxlo = np.array([float(v) for v in boxlo], dtype=np.float64)
        self.boxhi = np.array([float(v) for v in boxhi], dtype=np.float64)
        self.prd = self.boxhi - self.boxlo
        self.periodicity = np.array(
            [1 if b == "p" else 0 for b in boundary], dtype=np.int32
        )
        self._sync()

    def _sync(self) -> None:
        """Push the host-side box onto the device."""
        self.boxlo_f[None] = ti.Vector(self.boxlo.tolist())
        self.boxhi_f[None] = ti.Vector(self.boxhi.tolist())
        self.prd_f[None] = ti.Vector(self.prd.tolist())
        self.periodicity_f[None] = ti.Vector(self.periodicity.tolist())

    @property
    def volume(self) -> float:
        """Volume of the domain box."""
        return float(self.prd[0] * self.prd[1] * self.prd[2])

    def set_box(
        self,
        boxlo: list[float] | tuple[float, float, float],
        boxhi: list[float] | tuple[float, float, float],
    ) -> None:
        """Update domain bounds, on the host and on the device."""
        self.boxlo = np.array([float(v) for v in boxlo], dtype=np.float64)
        self.boxhi = np.array([float(v) for v in boxhi], dtype=np.float64)
        self.prd = self.boxhi - self.boxlo
        self._sync()

    def set_boundary(self, boundary: list[str] | tuple[str, str, str]) -> None:
        """Update boundary periodicity."""
        self.periodicity = np.array(
            [1 if b == "p" else 0 for b in boundary], dtype=np.int32
        )
        self._sync()

    @property
    def pbc_x(self) -> bool:
        return bool(self.periodicity[0])

    @property
    def pbc_y(self) -> bool:
        return bool(self.periodicity[1])

    @property
    def pbc_z(self) -> bool:
        return bool(self.periodicity[2])

    @ti.func
    def minimum_image(self, dx):
        """
        Apply minimum image convention for periodic boundaries.

        Closed form rather than a `while` loop: the loop form spins forever if a
        coordinate ever becomes non-finite, turning a diverging simulation into
        a hang instead of a visible NaN.
        """
        prd = self.prd_f[None]
        per = self.periodicity_f[None]
        for dim in ti.static(range(3)):
            if per[dim] == 1:
                dx[dim] -= prd[dim] * ti.round(dx[dim] / prd[dim])
        return dx

    @ti.func
    def pbc_wrap(self, x):
        """Wrap coordinates into [boxlo, boxhi), as LAMMPS Domain::pbc() does."""
        lo = self.boxlo_f[None]
        prd = self.prd_f[None]
        per = self.periodicity_f[None]
        for dim in ti.static(range(3)):
            if per[dim] == 1:
                s = (x[dim] - lo[dim]) / prd[dim]
                x[dim] = lo[dim] + (s - ti.floor(s)) * prd[dim]
        return x

    @ti.kernel
    def _pbc_kernel(self, nlocal: ti.i32, x: ti.template()):
        for i in range(nlocal):
            x[i] = self.pbc_wrap(x[i])

    def pbc(self, atom: Any) -> None:
        """Apply periodic boundary conditions to all atom positions."""
        self._pbc_kernel(atom.nlocal, atom.x)
