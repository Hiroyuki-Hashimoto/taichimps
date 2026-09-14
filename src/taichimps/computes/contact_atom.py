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
            xi = x[i]
            ri = radius[i]
            n_i = num_neigh[i]
            cnt = 0
            for k in range(n_i):
                j = neighbors[i, k]
                xj = x[j]
                rj = radius[j]
                d = domain.minimum_image(xi - xj)
                rsq = d.dot(d)
                radsum = ri + rj
                if rsq < radsum * radsum and rsq > 1e-18:
                    cnt += 1
            self.contact_count[i] = cnt

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
