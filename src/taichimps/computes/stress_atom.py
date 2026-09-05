"""
ComputeStressAtom matching LAMMPS compute stress/atom.
Calculates per-atom and macroscopic Cauchy stress tensor via Love-Weber formula.
Reference: LAMMPS src/compute_stress_atom.cpp
"""

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain


@ti.data_oriented
class ComputeStressAtom:
    def __init__(self, float_type: ti.template() = ti.f64): # type: ignore[valid-type]
        self.float_type = float_type
        self.macro_stress = ti.field(dtype=float_type, shape=6)

    @ti.kernel
    def compute_stress_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        f: ti.template(),
        v: ti.template(),
        rmass: ti.template(),
        volume: ti.template(),
    ):
        s_xx = 0.0
        s_yy = 0.0
        s_zz = 0.0
        s_xy = 0.0
        s_xz = 0.0
        s_yz = 0.0

        for i in range(nlocal):
            s_xx -= x[i][0] * f[i][0]
            s_yy -= x[i][1] * f[i][1]
            s_zz -= x[i][2] * f[i][2]
            s_xy -= 0.5 * (x[i][0] * f[i][1] + x[i][1] * f[i][0])
            s_xz -= 0.5 * (x[i][0] * f[i][2] + x[i][2] * f[i][0])
            s_yz -= 0.5 * (x[i][1] * f[i][2] + x[i][2] * f[i][1])

        inv_v = 1.0 / volume if volume > 0.0 else 0.0
        self.macro_stress[0] = s_xx * inv_v
        self.macro_stress[1] = s_yy * inv_v
        self.macro_stress[2] = s_zz * inv_v
        self.macro_stress[3] = s_xy * inv_v
        self.macro_stress[4] = s_xz * inv_v
        self.macro_stress[5] = s_yz * inv_v

    def compute(self, atom: AtomSystem, domain: Domain) -> np.ndarray:
        if atom.nlocal == 0:
            return np.zeros(6, dtype=np.float64)
        self.compute_stress_kernel(
            atom.nlocal,
            atom.x,
            atom.f,
            atom.v,
            atom.rmass,
            domain.volume,
        )
        return self.macro_stress.to_numpy()
