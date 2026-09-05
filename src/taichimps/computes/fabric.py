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
        prd: ti.template(),
        periodicity: ti.template(),
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
            domain.prd,
            domain.periodicity,
        )
        return self.phi.to_numpy(), int(self.num_contacts[None])
