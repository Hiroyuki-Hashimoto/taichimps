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
        prd_f: ti.template(),
        periodicity_f: ti.template(),
    ):
        # Zero-dimensional fields rather than Python vectors, so the current
        # box is read at launch instead of being baked in at compile time.
        prd = prd_f[None]
        periodicity = periodicity_f[None]
        for i in range(nlocal):
            xi = x[i]
            ri = radius[i]
            n_i = num_neigh[i]
            cnt = 0
            for k in range(n_i):
                j = neighbors[i, k]
                xj = x[j]
                rj = radius[j]
                dx = xi[0] - xj[0]
                dy = xi[1] - xj[1]
                dz = xi[2] - xj[2]
                if periodicity[0] and ti.abs(dx) > 0.5 * prd[0]:
                    dx -= ti.math.sign(dx) * prd[0]
                if periodicity[1] and ti.abs(dy) > 0.5 * prd[1]:
                    dy -= ti.math.sign(dy) * prd[1]
                if periodicity[2] and ti.abs(dz) > 0.5 * prd[2]:
                    dz -= ti.math.sign(dz) * prd[2]
                rsq = dx * dx + dy * dy + dz * dz
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
            domain.prd_f,
            domain.periodicity_f,
        )
        return self.contact_count.to_numpy()[:atom.nlocal]
