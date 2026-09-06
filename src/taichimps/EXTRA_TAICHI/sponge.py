"""
WinSponge / FixSponge: Windowed Non-reflecting Absorbing Boundary Layer.

Eliminates unwanted boundary reflections for wave propagation simulations (P, S, Rayleigh, Love)
in DEM by applying a smooth polynomial or Hann-windowed artificial viscous damping layer.

Reference:
- Lysmer, J., & Kuhlemeyer, R. L. (1969). Finite dynamic model for infinite media. J. Eng. Mech. Div.
- Israeli, M., & Orszag, S. A. (1981). Approximation of radiation boundary conditions. J. Comput. Phys.
"""

from collections.abc import Sequence
from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


@ti.data_oriented
class WinSponge(Fix):
    """Windowed Sponge absorbing boundary layer for granular DEM.

    Applies a spatially ramping viscous damping force F_i = -eta(d) * m_i * v_i
    towards specified outer domain boundaries.
    """

    def __init__(
        self,
        domain: Domain,
        float_type: Any = ti.f64,
        boundaries: Sequence[str] | None = None,
        thickness: float = 1.0,
        eta_max: float = 100.0,
        power: float = 2.0,
        window: str = "polynomial",
    ) -> None:
        super().__init__(domain=domain, float_type=float_type)
        if boundaries is None:
            boundaries = ("xlo", "xhi", "ylo", "yhi", "zlo")
        self.boundaries = [b.lower() for b in boundaries]
        self.thickness = float(thickness)
        self.eta_max = float(eta_max)
        self.power = float(power)
        self.window = window.lower()

        # Flags for active boundaries: [xlo, xhi, ylo, yhi, zlo, zhi]
        self.flags = [
            1 if "xlo" in self.boundaries else 0,
            1 if "xhi" in self.boundaries else 0,
            1 if "ylo" in self.boundaries else 0,
            1 if "yhi" in self.boundaries else 0,
            1 if "zlo" in self.boundaries else 0,
            1 if "zhi" in self.boundaries else 0,
        ]

    @ti.kernel
    def sponge_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        v: ti.template(),
        f: ti.template(),
        rmass: ti.template(),
        boxlo_x: ti.template(),
        boxlo_y: ti.template(),
        boxlo_z: ti.template(),
        boxhi_x: ti.template(),
        boxhi_y: ti.template(),
        boxhi_z: ti.template(),
        thickness: ti.template(),
        eta_max: ti.template(),
        power: ti.template(),
        flag_xlo: ti.i32,
        flag_xhi: ti.i32,
        flag_ylo: ti.i32,
        flag_yhi: ti.i32,
        flag_zlo: ti.i32,
        flag_zhi: ti.i32,
    ):
        for i in range(nlocal):
            px = x[i][0]
            py = x[i][1]
            pz = x[i][2]

            max_d_norm = 0.0

            # xlo
            if flag_xlo != 0:
                dist = px - boxlo_x
                if dist < thickness and dist >= 0.0:
                    norm = (thickness - dist) / thickness
                    max_d_norm = max(max_d_norm, norm)
            # xhi
            if flag_xhi != 0:
                dist = boxhi_x - px
                if dist < thickness and dist >= 0.0:
                    norm = (thickness - dist) / thickness
                    max_d_norm = max(max_d_norm, norm)

            # ylo
            if flag_ylo != 0:
                dist = py - boxlo_y
                if dist < thickness and dist >= 0.0:
                    norm = (thickness - dist) / thickness
                    max_d_norm = max(max_d_norm, norm)
            # yhi
            if flag_yhi != 0:
                dist = boxhi_y - py
                if dist < thickness and dist >= 0.0:
                    norm = (thickness - dist) / thickness
                    max_d_norm = max(max_d_norm, norm)

            # zlo
            if flag_zlo != 0:
                dist = pz - boxlo_z
                if dist < thickness and dist >= 0.0:
                    norm = (thickness - dist) / thickness
                    max_d_norm = max(max_d_norm, norm)
            # zhi
            if flag_zhi != 0:
                dist = boxhi_z - pz
                if dist < thickness and dist >= 0.0:
                    norm = (thickness - dist) / thickness
                    max_d_norm = max(max_d_norm, norm)

            if max_d_norm > 0.0:
                # Polynomial ramp: w = norm^power
                w = max_d_norm**power
                m = rmass[i]
                coeff = eta_max * w * m

                f[i][0] -= coeff * v[i][0]
                f[i][1] -= coeff * v[i][1]
                f[i][2] -= coeff * v[i][2]

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        if atom.nlocal == 0 or self.domain is None:
            return

        self.sponge_kernel(
            atom.nlocal,
            atom.x,
            atom.v,
            atom.f,
            atom.rmass,
            self.domain.boxlo[0],
            self.domain.boxlo[1],
            self.domain.boxlo[2],
            self.domain.boxhi[0],
            self.domain.boxhi[1],
            self.domain.boxhi[2],
            self.thickness,
            self.eta_max,
            self.power,
            self.flags[0],
            self.flags[1],
            self.flags[2],
            self.flags[3],
            self.flags[4],
            self.flags[5],
        )


# Alias for LAMMPS style naming
FixSponge = WinSponge
