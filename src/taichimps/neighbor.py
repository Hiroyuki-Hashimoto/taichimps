"""
Spatial grid / neighbor list for granular contact search.
Reference: LAMMPS src/neighbor.cpp, src/npair_bin.cpp
License: GPL v2 compatible / taichimps MIT reimplementation
"""

from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain


@ti.data_oriented
class NeighborList:
    """
    Spatial binning neighbor list for spherical granular particles.
    Maintains a contact pair list and contact history indices.
    """

    def __init__(
        self,
        domain: Domain,
        skin: float = 0.001,
        max_atoms: int = 100000,
        max_neighbors_per_atom: int = 64,
        max_neighbors: int | None = None,
        float_type: Any = ti.f64,
    ) -> None:
        self.domain = domain
        self.skin = skin
        self.max_atoms = max_atoms
        if max_neighbors is not None:
            self.max_neighbors_per_atom = max_neighbors
        else:
            self.max_neighbors_per_atom = max_neighbors_per_atom
        self.float_type = float_type

        # Neighbor list arrays
        self.num_neighbors = ti.field(dtype=ti.i32, shape=max_atoms)
        self.neighbors = ti.field(
            dtype=ti.i32, shape=(max_atoms, max_neighbors_per_atom)
        )

        # Displacement check for skin / neigh_modify
        self.x0 = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.max_displacement_sq = ti.field(dtype=float_type, shape=())

        # Parameters for neigh_modify (LAMMPS default: delay 0 every 1 check yes)
        self.delay: int = 0
        self.every: int = 1
        self.check: bool = True
        self.last_build_step: int = -1
        self.current_step: int = 0
        self.has_built: bool = False
        self.build_count: int = 0

        # Spatial grid parameters
        # Cell size will be set dynamically based on max particle diameter + skin
        self.grid_dim = ti.Vector.field(3, dtype=ti.i32, shape=())
        self.cell_size = ti.Vector.field(3, dtype=float_type, shape=())

        # Grid linked-list data structures
        self.max_grid_cells: int = 200000
        self.grid_head = ti.field(dtype=ti.i32, shape=self.max_grid_cells)
        self.grid_next = ti.field(dtype=ti.i32, shape=max_atoms)

    def setup_grid(self, max_cutoff: float) -> None:
        """Setup grid dimensions given maximum cutoff distance (2 * r_max + skin)."""
        cut = max_cutoff + self.skin
        if cut <= 0:
            cut = 1.0

        prd = [
            float(self.domain.prd[0]),
            float(self.domain.prd[1]),
            float(self.domain.prd[2]),
        ]
        gx = max(1, int(np.floor(prd[0] / cut)))
        gy = max(1, int(np.floor(prd[1] / cut)))
        gz = max(1, int(np.floor(prd[2] / cut)))

        total_cells = gx * gy * gz
        if total_cells > self.max_grid_cells:
            # Scale down grid resolution to fit memory
            factor = (self.max_grid_cells / total_cells) ** (1.0 / 3.0)
            gx = max(1, int(gx * factor))
            gy = max(1, int(gy * factor))
            gz = max(1, int(gz * factor))

        self.grid_dim[None] = ti.Vector([gx, gy, gz])
        self.cell_size[None] = ti.Vector(
            [prd[0] / gx, prd[1] / gy, prd[2] / gz], dt=self.float_type
        )

    @ti.func
    def get_cell_coord(self, pos):
        gdim = self.grid_dim[None]
        csize = self.cell_size[None]
        cx = ti.cast(ti.floor((pos[0] - self.domain.boxlo[0]) / csize[0]), ti.i32)
        cy = ti.cast(ti.floor((pos[1] - self.domain.boxlo[1]) / csize[1]), ti.i32)
        cz = ti.cast(ti.floor((pos[2] - self.domain.boxlo[2]) / csize[2]), ti.i32)

        # Clamp inside grid boundaries
        cx = ti.max(0, ti.min(cx, gdim[0] - 1))
        cy = ti.max(0, ti.min(cy, gdim[1] - 1))
        cz = ti.max(0, ti.min(cz, gdim[2] - 1))
        return ti.Vector([cx, cy, cz])

    @ti.kernel
    def store_x0(self, nlocal: ti.i32, x: ti.template()):
        """Store reference particle positions for displacement checking."""
        for i in range(nlocal):
            self.x0[i] = x[i]

    @ti.kernel
    def check_displacement(self, nlocal: ti.i32, x: ti.template()):
        """Find max displacement squared among all local particles relative to x0."""
        self.max_displacement_sq[None] = 0.0
        for i in range(nlocal):
            diff = x[i] - self.x0[i]
            dsq = diff.dot(diff)
            ti.atomic_max(self.max_displacement_sq[None], dsq)

    @ti.kernel
    def build_grid(self, nlocal: ti.i32, x: ti.template()):
        gdim = self.grid_dim[None]
        gx, gy, gz = gdim[0], gdim[1], gdim[2]
        gxy = gx * gy
        num_cells = gxy * gz
        for c in range(num_cells):
            self.grid_head[c] = -1

        for i in range(nlocal):
            cell = self.get_cell_coord(x[i])
            c_idx = cell[0] + cell[1] * gx + cell[2] * gxy
            old_head = self.grid_head[c_idx]
            self.grid_next[i] = old_head
            self.grid_head[c_idx] = i

    @ti.kernel
    def build_neighbor_list(
        self,
        nlocal: ti.i32,
        x: ti.template(),
        radius: ti.template(),
    ):
        """
        Build neighbor list using spatial binning.
        Only stores pairs (i, j) with i < j (half neighbor list).
        """
        gdim = self.grid_dim[None]
        gx, gy, gz = gdim[0], gdim[1], gdim[2]
        gxy = gx * gy
        px, py, pz = (
            self.domain.periodicity[0],
            self.domain.periodicity[1],
            self.domain.periodicity[2],
        )

        for i in range(nlocal):
            self.num_neighbors[i] = 0
            pos_i = x[i]
            r_i = radius[i]
            cell_i = self.get_cell_coord(pos_i)
            cx, cy, cz = cell_i[0], cell_i[1], cell_i[2]

            count = 0
            for dx, dy, dz in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                nx = cx + dx
                ny = cy + dy
                nz = cz + dz

                # Handle boundary wrapping
                if px == 1:
                    nx = (nx % gx + gx) % gx
                if py == 1:
                    ny = (ny % gy + gy) % gy
                if pz == 1:
                    nz = (nz % gz + gz) % gz

                if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz:
                    c_idx = nx + ny * gx + nz * gxy
                    j = self.grid_head[c_idx]
                    while j != -1 and count < self.max_neighbors_per_atom:
                        if j > i:
                            dpos = self.domain.minimum_image(pos_i - x[j])
                            rsq = dpos.dot(dpos)
                            rad_sum = r_i + radius[j] + self.skin
                            if rsq < rad_sum * rad_sum:
                                self.neighbors[i, count] = j
                                count += 1
                        j = self.grid_next[j]

            self.num_neighbors[i] = count

    def decide(self, atom: AtomSystem, timestep: int | None = None) -> bool:
        """Determine if neighbor list rebuild is required according to LAMMPS criteria."""
        if not self.has_built:
            return True

        if timestep is not None:
            self.current_step = timestep
        else:
            self.current_step += 1

        steps_since_build = self.current_step - self.last_build_step

        # delay option: don't build until delay steps have elapsed since last build
        if steps_since_build < self.delay:
            return False

        # every option: only build every N steps
        if steps_since_build % self.every != 0:
            return False

        # check option: if check is False, rebuild on every step that satisfies delay/every
        if not self.check:
            return True

        # Skin displacement check: max(|x - x0|) > 0.5 * skin -> dsq > (0.5 * skin)^2
        if atom.nlocal == 0:
            return False

        self.check_displacement(atom.nlocal, atom.x)
        max_dsq = float(self.max_displacement_sq[None])
        trigger_sq = 0.25 * self.skin * self.skin
        return max_dsq > trigger_sq

    def check_and_build(
        self, atom: AtomSystem, timestep: int | None = None
    ) -> bool:
        """Check criteria and build neighbor list if needed. Returns True if rebuilt."""
        if self.decide(atom, timestep):
            self.build(atom)
            return True
        return False

    def build(self, atom: AtomSystem) -> None:
        """Execute neighbor list construction."""
        if atom.nlocal == 0:
            return
        # Calculate max cutoff if not initialized
        r_np = atom.radius.to_numpy()[: atom.nlocal]
        max_r = float(np.max(r_np)) if len(r_np) > 0 else 1.0
        self.setup_grid(2.0 * max_r)
        self.build_grid(atom.nlocal, atom.x)
        self.build_neighbor_list(atom.nlocal, atom.x, atom.radius)
        self.store_x0(atom.nlocal, atom.x)
        self.has_built = True
        self.build_count += 1
        self.last_build_step = self.current_step
