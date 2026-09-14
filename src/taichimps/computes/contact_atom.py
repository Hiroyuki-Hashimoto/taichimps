"""
ComputeContactAtom matching LAMMPS compute contact/atom.
Calculates per-atom contact count (coordination number).
Reference: LAMMPS src/compute_contact_atom.cpp
"""

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.neighbor import NeighborList


@ti.data_oriented
class ComputeContactAtom:
    def __init__(self, max_atoms: int = 2000000):
        self.contact_count = ti.field(dtype=ti.i32, shape=max_atoms)

    @ti.kernel
    def compute_kernel(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        radius: ti.template(),
        num_neigh: ti.template(),
        neighbors: ti.template(),
        domain: ti.template(),
    ):
        # Delegate the minimum image to Domain so a tilted cell is handled the
        # same way here as in the pair styles; the hand-rolled per-axis version
        # this replaced had no tilt corrections at all.
        for i in range(nlocal):
            self.contact_count[i] = 0

        # The neighbor list is a half list (only j > i), while LAMMPS builds
        # this compute on a full list where every contact is seen from both
        # sides. Each contact therefore has to be tallied to both partners;
        # counting it only for the lower index returned exactly half the
        # coordination number.
        for i in range(nlocal):
            xi = x[i]
            ri = radius[i]
            n_i = num_neigh[i]
            for k in range(n_i):
                j = neighbors[i, k]
                d = domain.minimum_image(xi - x[j])
                rsq = d.dot(d)
                radsum = ri + radius[j]
                if rsq < radsum * radsum and rsq > 1e-18:
                    ti.atomic_add(self.contact_count[i], 1)
                    ti.atomic_add(self.contact_count[j], 1)

    def compute(self, atom: AtomSystem, domain: Domain, neighbor: NeighborList) -> np.ndarray:
        if atom.nlocal == 0:
            return np.zeros(0, dtype=np.int32)
        self.compute_kernel(
            atom.nlocal,
            atom.x,
            atom.radius,
            neighbor.num_neighbors,
            neighbor.neighbors,
            domain,
        )
        return self.contact_count.to_numpy()[:atom.nlocal]
