"""
Simulation domain and periodic boundary conditions.
Reference: LAMMPS src/domain.cpp
"""

from typing import Any

import taichi as ti


@ti.data_oriented
class Domain:
    """
    Orthogonal simulation domain [boxlo, boxhi] with boundary conditions.
    bcond: 'p' (periodic), 'f' (fixed/non-periodic).
    e.g. boundary=['p', 'p', 'f'] for x, y periodic, z fixed.
    """

    def __init__(
        self,
        boxlo: list[float] | tuple[float, float, float],
        boxhi: list[float] | tuple[float, float, float],
        boundary: list[str] | tuple[str, str, str] = ("p", "p", "p"),
        float_type: Any = ti.f64,
    ) -> None:
        self.boxlo = ti.Vector([float(boxlo[0]), float(boxlo[1]), float(boxlo[2])], dt=float_type)
        self.boxhi = ti.Vector([float(boxhi[0]), float(boxhi[1]), float(boxhi[2])], dt=float_type)
        self.prd = ti.Vector(
            [
                float(boxhi[0] - boxlo[0]),
                float(boxhi[1] - boxlo[1]),
                float(boxhi[2] - boxlo[2]),
            ],
            dt=float_type,
        )
        self.periodicity = ti.Vector(
            [
                1 if boundary[0] == "p" else 0,
                1 if boundary[1] == "p" else 0,
                1 if boundary[2] == "p" else 0,
            ],
            dt=ti.i32,
        )

    @property
    def volume(self) -> float:
        """Volume of the domain box."""
        return float(self.prd[0] * self.prd[1] * self.prd[2])

    def set_box(self, boxlo: list[float] | tuple[float, float, float], boxhi: list[float] | tuple[float, float, float]) -> None:
        """Update domain bounds."""
        for dim in range(3):
            self.boxlo[dim] = float(boxlo[dim])
            self.boxhi[dim] = float(boxhi[dim])
            self.prd[dim] = float(boxhi[dim] - boxlo[dim])

    def set_boundary(self, boundary: list[str] | tuple[str, str, str]) -> None:
        """Update boundary periodicity."""
        for dim in range(3):
            self.periodicity[dim] = 1 if boundary[dim] == "p" else 0

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
        for dim in ti.static(range(3)):
            if self.periodicity[dim] == 1:
                dx[dim] -= self.prd[dim] * ti.round(dx[dim] / self.prd[dim])
        return dx

    @ti.func
    def pbc_wrap(self, x):
        """Wrap coordinates into [boxlo, boxhi), as LAMMPS Domain::pbc() does."""
        for dim in ti.static(range(3)):
            if self.periodicity[dim] == 1:
                s = (x[dim] - self.boxlo[dim]) / self.prd[dim]
                x[dim] = self.boxlo[dim] + (s - ti.floor(s)) * self.prd[dim]
        return x

    @ti.kernel
    def _pbc_kernel(self, nlocal: ti.i32, x: ti.template()):
        for i in range(nlocal):
            x[i] = self.pbc_wrap(x[i])

    def pbc(self, atom: Any) -> None:
        """Apply periodic boundary conditions to all atom positions."""
        self._pbc_kernel(atom.nlocal, atom.x)

