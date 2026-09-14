"""
ComputeFabric matching LAMMPS compute fabric (GRANULAR package).
Calculates fabric tensors Phi_ij for geotechnical analysis:
- Contact normal fabric: Phi_ij = (1 / Nc) * sum(n_i * n_j)
- Average coordination number
Reference: LAMMPS src/GRANULAR/compute_fabric.cpp
"""

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.neighbor import NeighborList


@ti.data_oriented
class ComputeFabric:
    def __init__(self, float_type: ti.template() = ti.f64): # type: ignore[valid-type]
        self.float_type = float_type
        self.phi = ti.field(dtype=float_type, shape=6) # xx, yy, zz, xy, xz, yz
        self.num_contacts = ti.field(dtype=ti.i32, shape=())

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
        self.num_contacts[None] = 0
        s_xx = 0.0
        s_yy = 0.0
        s_zz = 0.0
        s_xy = 0.0
        s_xz = 0.0
        s_yz = 0.0
        n_c = 0

        for i in range(nlocal):
            xi = x[i]
            ri = radius[i]
            n_i = num_neigh[i]
            for k in range(n_i):
                j = neighbors[i, k]
                if j > i:
                    xj = x[j]
                    rj = radius[j]
                    # Delegate the minimum image to Domain so a tilted cell is
                    # handled the same way here as in the pair styles; the
                    # hand-rolled per-axis version this replaced had no tilt
                    # corrections at all.
                    d = domain.minimum_image(xi - xj)
                    dx, dy, dz = d[0], d[1], d[2]
                    rsq = d.dot(d)
                    radsum = ri + rj
                    if rsq < radsum * radsum and rsq > 1e-18:
                        r = ti.sqrt(rsq)
                        nx = dx / r
                        ny = dy / r
                        nz = dz / r
                        s_xx += nx * nx
                        s_yy += ny * ny
                        s_zz += nz * nz
                        s_xy += nx * ny
                        s_xz += nx * nz
                        s_yz += ny * nz
                        n_c += 1

        self.num_contacts[None] = n_c
        if n_c > 0:
            inv_nc = 1.0 / ti.cast(n_c, self.float_type)
            self.phi[0] = s_xx * inv_nc
            self.phi[1] = s_yy * inv_nc
            self.phi[2] = s_zz * inv_nc
            self.phi[3] = s_xy * inv_nc
            self.phi[4] = s_xz * inv_nc
            self.phi[5] = s_yz * inv_nc
        else:
            for m in range(6):
                self.phi[m] = 0.0

    def compute(self, atom: AtomSystem, domain: Domain, neighbor: NeighborList) -> tuple[np.ndarray, int]:
        if atom.nlocal == 0:
            return np.zeros(6, dtype=np.float64), 0
        self.compute_kernel(
            atom.nlocal,
            atom.x,
            atom.radius,
            neighbor.num_neighbors,
            neighbor.neighbors,
            domain,
        )
        return self.phi.to_numpy(), int(self.num_contacts[None])
