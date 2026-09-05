"""
Thermodynamic property compute routines.
Reference: LAMMPS src/compute_temp.cpp, src/compute_pressure.cpp, src/compute_ke.cpp
"""

from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain


@ti.data_oriented
class Computes:
    """Computes global thermodynamic scalar and tensor quantities."""

    def __init__(self, float_type: Any = ti.f64) -> None:
        self.float_type = float_type
        self.ke_trans_val = ti.field(dtype=float_type, shape=())
        self.ke_rot_val = ti.field(dtype=float_type, shape=())
        self.virial_tensor = ti.field(dtype=float_type, shape=6)

    @ti.kernel
    def compute_ke_kernel(
        self,
        nlocal: ti.i32,
        v: ti.template(),
        omega: ti.template(),
        radius: ti.template(),
        rmass: ti.template(),
    ):
        self.ke_trans_val[None] = 0.0
        self.ke_rot_val[None] = 0.0
        for i in range(nlocal):
            vsq = v[i].dot(v[i])
            self.ke_trans_val[None] += 0.5 * rmass[i] * vsq

            i_moment = 0.4 * rmass[i] * radius[i] * radius[i]
            wsq = omega[i].dot(omega[i])
            self.ke_rot_val[None] += 0.5 * i_moment * wsq

    @ti.kernel
    def compute_virial_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        f: ti.template(),
        v: ti.template(),
        rmass: ti.template(),
    ):
        for k in range(6):
            self.virial_tensor[k] = 0.0

        for i in range(nlocal):
            # Kinetic part: m * v_a * v_b
            mvx2 = rmass[i] * v[i][0] * v[i][0]
            mvy2 = rmass[i] * v[i][1] * v[i][1]
            mvz2 = rmass[i] * v[i][2] * v[i][2]
            mvxy = rmass[i] * v[i][0] * v[i][1]
            mvxz = rmass[i] * v[i][0] * v[i][2]
            mvyz = rmass[i] * v[i][1] * v[i][2]

            # Virial part from total forces: x_a * f_b
            # Note: For pair interactions, sum x_i * f_i equals -0.5 sum r_ij * f_ij
            # In LAMMPS Virial convention: P_ab = (sum m v_a v_b + sum r_ab * f_ab) / V
            self.virial_tensor[0] += mvx2 + x[i][0] * f[i][0]
            self.virial_tensor[1] += mvy2 + x[i][1] * f[i][1]
            self.virial_tensor[2] += mvz2 + x[i][2] * f[i][2]
            self.virial_tensor[3] += mvxy + x[i][0] * f[i][1]
            self.virial_tensor[4] += mvxz + x[i][0] * f[i][2]
            self.virial_tensor[5] += mvyz + x[i][1] * f[i][2]

    def ke_trans(self, atom: AtomSystem) -> float:
        """Total translational kinetic energy."""
        if atom.nlocal == 0:
            return 0.0
        self.compute_ke_kernel(
            atom.nlocal,
            atom.v,
            atom.omega,
            atom.radius,
            atom.rmass,
        )
        return float(self.ke_trans_val[None])

    def ke_rot(self, atom: AtomSystem) -> float:
        """Total rotational kinetic energy."""
        if atom.nlocal == 0:
            return 0.0
        self.compute_ke_kernel(
            atom.nlocal,
            atom.v,
            atom.omega,
            atom.radius,
            atom.rmass,
        )
        return float(self.ke_rot_val[None])

    def ke_total(self, atom: AtomSystem) -> float:
        """Total kinetic energy (trans + rot)."""
        if atom.nlocal == 0:
            return 0.0
        self.compute_ke_kernel(
            atom.nlocal,
            atom.v,
            atom.omega,
            atom.radius,
            atom.rmass,
        )
        return float(self.ke_trans_val[None] + self.ke_rot_val[None])

    def compute_pressure_tensor(self, atom: AtomSystem, domain: Domain) -> np.ndarray:
        """Compute the 6 pressure tensor components: [Pxx, Pyy, Pzz, Pxy, Pxz, Pyz]."""
        if atom.nlocal == 0:
            return np.zeros(6, dtype=np.float64)
        self.compute_virial_kernel(
            atom.nlocal,
            atom.x,
            atom.f,
            atom.v,
            atom.rmass,
        )
        vol = domain.volume
        if vol <= 0.0:
            return np.zeros(6, dtype=np.float64)
        vir = self.virial_tensor.to_numpy()
        return vir / vol

    def compute_coordination_number(self, atom: AtomSystem, neighbor: Any) -> np.ndarray:
        """Compute coordination number (number of neighbors) per particle."""
        if atom.nlocal == 0:
            return np.zeros(0, dtype=np.int32)
        return neighbor.num_neighbors.to_numpy()[: atom.nlocal]
