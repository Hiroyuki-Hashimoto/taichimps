"""
Thermodynamic property compute routines.
Reference: LAMMPS src/compute_temp.cpp, src/compute_pressure.cpp, src/compute_ke.cpp
"""

import math
from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain

# LAMMPS `units si`: force->boltz
BOLTZMANN = 1.3806504e-23

# The virial reduction is a two-level tree: the first pass splits the atoms
# across this many partial sums, the second adds the partials up. Both passes
# cost about sqrt(N) when the split is 2*sqrt(N) wide, which is why it scales
# that way rather than being fixed -- too few chunks and the first pass is a
# long serial walk per thread, too many and the second is (six threads times
# chunks) of dependent adds, measured at 0.036 ms with 1024 chunks against
# 0.017 ms with 64 for the 1400-particle case.
MAX_VIRIAL_CHUNKS = 1024


def virial_chunks(nlocal: int) -> int:
    """How wide to split the virial reduction for this many atoms."""
    return int(min(MAX_VIRIAL_CHUNKS, max(64, 2.0 * math.sqrt(max(nlocal, 1)))))


@ti.data_oriented
class Computes:
    """Computes global thermodynamic scalar and tensor quantities."""

    def __init__(self, float_type: Any = ti.f64) -> None:
        self.float_type = float_type
        self.ke_trans_val = ti.field(dtype=float_type, shape=())
        self.ke_rot_val = ti.field(dtype=float_type, shape=())
        self.virial_tensor = ti.field(dtype=float_type, shape=6)
        # Per-chunk partial sums for the virial reduction. Each chunk owns its
        # slot, so nothing is contended and nothing has to be cleared first.
        self.virial_partial = ti.Vector.field(6, dtype=float_type, shape=MAX_VIRIAL_CHUNKS)

    @ti.kernel
    def compute_ke_kernel(
        self,
        nlocal: ti.template(),
        atom: ti.template(),
    ):
        self.ke_trans_val[None] = 0.0
        self.ke_rot_val[None] = 0.0
        for i in range(nlocal):
            vsq = atom.v[i].dot(atom.v[i])
            self.ke_trans_val[None] += 0.5 * atom.rmass[i] * vsq

            i_moment = 0.4 * atom.rmass[i] * atom.radius[i] * atom.radius[i]
            wsq = atom.omega[i].dot(atom.omega[i])
            self.ke_rot_val[None] += 0.5 * i_moment * wsq

    @ti.kernel
    def compute_virial_kernel(
        self,
        nlocal: ti.template(),
        atom: ti.template(),
        keflag: ti.i32,
        chunks: ti.template(),
    ):
        """
        Sum the pairwise virial tallied by the force kernels, plus optionally
        the kinetic term, into the 6-component tensor [xx, yy, zz, xy, xz, yz].

        The virial itself is accumulated per contact (0.5 * del_a * f_b into
        each partner) in the pair styles, exactly as LAMMPS does in
        Pair::ev_tally_xyz(). Summing sum(x_i . f_i) instead -- as this used to
        -- is not translation invariant under periodic boundaries, because
        taichimps has no ghost atoms to unwrap against.

        Done in two passes rather than by having every thread add into the same
        six globals. That form put a thousand threads on six addresses, and
        CUDA serialises atomics that collide, so it cost 0.026 ms per step --
        more than the whole rest of the step outside the contact kernel. It
        also needed the six globals cleared first, and a scalar statement at
        kernel scope compiles to its own serial GPU launch. Here each chunk
        owns a slot and simply writes it, so there is no contention and nothing
        to clear.
        """
        for c in range(chunks):
            acc = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dt=self.float_type)
            i = c
            while i < nlocal:
                acc += atom.virial[i]
                if keflag != 0:
                    m = atom.rmass[i]
                    acc[0] += m * atom.v[i][0] * atom.v[i][0]
                    acc[1] += m * atom.v[i][1] * atom.v[i][1]
                    acc[2] += m * atom.v[i][2] * atom.v[i][2]
                    acc[3] += m * atom.v[i][0] * atom.v[i][1]
                    acc[4] += m * atom.v[i][0] * atom.v[i][2]
                    acc[5] += m * atom.v[i][1] * atom.v[i][2]
                i += chunks
            self.virial_partial[c] = acc

        for k in range(6):
            total = 0.0
            for c in range(chunks):
                total += self.virial_partial[c][k]
            self.virial_tensor[k] = total

    def ke_trans(self, atom: AtomSystem) -> float:
        """Total translational kinetic energy."""
        if atom.nlocal == 0:
            return 0.0
        self.compute_ke_kernel(
            atom.nlocal,
            atom,
        )
        return float(self.ke_trans_val[None])

    def ke_rot(self, atom: AtomSystem) -> float:
        """Total rotational kinetic energy."""
        if atom.nlocal == 0:
            return 0.0
        self.compute_ke_kernel(
            atom.nlocal,
            atom,
        )
        return float(self.ke_rot_val[None])

    def ke_total(self, atom: AtomSystem) -> float:
        """Total kinetic energy (trans + rot)."""
        if atom.nlocal == 0:
            return 0.0
        self.compute_ke_kernel(
            atom.nlocal,
            atom,
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
            atom,
            1 if kinetic else 0,
            virial_chunks(atom.nlocal),
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
            float(np.sum(mv[:, 0] * atom.v[:, 0])),
            float(np.sum(mv[:, 1] * atom.v[:, 1])),
            float(np.sum(mv[:, 2] * atom.v[:, 2])),
            float(np.sum(mv[:, 0] * atom.v[:, 1])),
            float(np.sum(mv[:, 0] * atom.v[:, 2])),
            float(np.sum(mv[:, 1] * atom.v[:, 2])),
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
