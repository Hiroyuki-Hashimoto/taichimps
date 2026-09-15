"""
Thermodynamic property compute routines.
Reference: LAMMPS src/compute_temp.cpp, src/compute_pressure.cpp, src/compute_ke.cpp
"""

from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain

# LAMMPS `units si`: force->boltz
BOLTZMANN = 1.3806504e-23


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
        nlocal: ti.template(),
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
        nlocal: ti.template(),
        v: ti.template(),
        rmass: ti.template(),
        atom_virial: ti.template(),
        keflag: ti.i32,
    ):
        """
        Sum the pairwise virial tallied by the force kernels, plus optionally
        the kinetic term, into the 6-component tensor [xx, yy, zz, xy, xz, yz].

        The virial itself is accumulated per contact (0.5 * del_a * f_b into
        each partner) in the pair styles, exactly as LAMMPS does in
        Pair::ev_tally_xyz(). Summing sum(x_i . f_i) instead -- as this used to
        -- is not translation invariant under periodic boundaries, because
        taichimps has no ghost atoms to unwrap against.
        """
        # `nlocal` is a ti.template(), i.e. baked into the compiled kernel,
        # not passed at launch. A loop whose bound is a runtime argument makes
        # Taichi emit a second, serial GPU launch ahead of the parallel one
        # just to establish the range; measured at 6.9 us per call on CUDA,
        # which at this system size is a third of the kernel's own cost.
        # Taichi recompiles per distinct value, so a run whose particle count
        # never changes compiles this once.
        for k in ti.static(range(6)):
            self.virial_tensor[k] = 0.0

        for i in range(nlocal):
            vir = atom_virial[i]
            for k in ti.static(range(6)):
                self.virial_tensor[k] += vir[k]

            if keflag != 0:
                m = rmass[i]
                self.virial_tensor[0] += m * v[i][0] * v[i][0]
                self.virial_tensor[1] += m * v[i][1] * v[i][1]
                self.virial_tensor[2] += m * v[i][2] * v[i][2]
                self.virial_tensor[3] += m * v[i][0] * v[i][1]
                self.virial_tensor[4] += m * v[i][0] * v[i][2]
                self.virial_tensor[5] += m * v[i][1] * v[i][2]

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

    def compute_pressure_tensor(
        self,
        atom: AtomSystem,
        domain: Domain,
        kinetic: bool = True,
    ) -> np.ndarray:
        """
        The 6 pressure tensor components [Pxx, Pyy, Pzz, Pxy, Pxz, Pyz].

        `kinetic=False` corresponds to the LAMMPS `compute pressure NULL pair`
        form, where no temperature compute is given and only the pair virial
        contributes. Compression gives a positive pressure, as in LAMMPS.
        """
        if atom.nlocal == 0:
            return np.zeros(6, dtype=np.float64)
        # Costs nothing unless a kernel has moved the box on without telling
        # the host, which only happens on the device servo path -- and there
        # this is reached only when something asks for the pressure, i.e. on
        # output steps.
        domain.pull()
        vol = domain.volume
        if vol <= 0.0:
            return np.zeros(6, dtype=np.float64)
        self.compute_virial_kernel(
            atom.nlocal,
            atom.v,
            atom.rmass,
            atom.virial,
            1 if kinetic else 0,
        )
        return self.virial_tensor.to_numpy() / vol

    def kinetic_tensor(self, atom: AtomSystem) -> np.ndarray:
        """
        The kinetic energy tensor sum(m * v_a * v_b), as `compute temp` reports it.

        Note the LAMMPS convention: no factor of 1/2, and in `units si`
        mvv2e is 1 so no unit conversion either.
        """
        if atom.nlocal == 0:
            return np.zeros(6, dtype=np.float64)
        n = atom.nlocal
        v = atom.v.to_numpy()[:n]
        m = atom.rmass.to_numpy()[:n]
        mv = m[:, None] * v
        return np.array([
            float(np.sum(mv[:, 0] * v[:, 0])),
            float(np.sum(mv[:, 1] * v[:, 1])),
            float(np.sum(mv[:, 2] * v[:, 2])),
            float(np.sum(mv[:, 0] * v[:, 1])),
            float(np.sum(mv[:, 0] * v[:, 2])),
            float(np.sum(mv[:, 1] * v[:, 2])),
        ])

    def temperature(self, atom: AtomSystem) -> float:
        """
        Instantaneous temperature, as `compute temp` reports it.

        t = sum(m v^2) / (dof * kB) with dof = 3N - 3, matching
        ComputeTemp::dof_compute() for point-like degrees of freedom (the
        rotational degrees of freedom of a sphere belong to compute
        temp/sphere, not to this one).
        """
        if atom.nlocal == 0:
            return 0.0
        dof = 3.0 * atom.nlocal - 3.0
        if dof <= 0.0:
            return 0.0
        tensor = self.kinetic_tensor(atom)
        return float(np.sum(tensor[:3]) / (dof * BOLTZMANN))

    def compute_coordination_number(self, atom: AtomSystem, neighbor: Any) -> np.ndarray:
        """Compute coordination number (number of neighbors) per particle."""
        if atom.nlocal == 0:
            return np.zeros(0, dtype=np.int32)
        return neighbor.num_neighbors.to_numpy()[: atom.nlocal]
