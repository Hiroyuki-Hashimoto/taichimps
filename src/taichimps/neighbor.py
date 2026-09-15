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

        # Neighbor list arrays.  Size from self.max_neighbors_per_atom, which
        # accounts for the `max_neighbors` alias; using the raw argument here
        # would under-allocate whenever the alias asked for a larger list and
        # the build kernel would write past the end of the field.
        self.num_neighbors = ti.field(dtype=ti.i32, shape=max_atoms)
        self.neighbors = ti.field(
            dtype=ti.i32, shape=(max_atoms, self.max_neighbors_per_atom)
        )
        # Set when a candidate pair could not be stored because the per-atom
        # list was full.  LAMMPS grows its pages instead; we cannot resize a
        # Taichi field mid-run, so this is reported as an error rather than
        # silently dropping contacts.
        self.overflow = ti.field(dtype=ti.i32, shape=())

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

        # Cell lists, held as a counting sort rather than as a linked list.
        #
        # The linked-list form this replaces built each bin by
        #     old = head[c]; next[i] = old; head[c] = i
        # which is a read-modify-write on head[c] with no atomic around it.
        # Taichi parallelises that loop, so two particles landing in the same
        # bin race and one of them is dropped from the bin entirely -- on CUDA
        # this lost 47% of the contacts of the 1400-particle FCC packing, which
        # is not detectable from the outside as anything but wrong forces.
        # Taichi offers no atomic exchange to repair it with, and a counting
        # sort is the better structure anyway: the members of a bin end up
        # contiguous in memory, so the pair search reads them as a run instead
        # of chasing pointers through the whole particle array.
        self.max_grid_cells: int = 200000
        self.cell_of = ti.field(dtype=ti.i32, shape=max_atoms)
        self.cell_count = ti.field(dtype=ti.i32, shape=self.max_grid_cells + 1)
        self.cell_start = ti.field(dtype=ti.i32, shape=self.max_grid_cells + 1)
        self.cell_fill = ti.field(dtype=ti.i32, shape=self.max_grid_cells)
        self.cell_particles = ti.field(dtype=ti.i32, shape=max_atoms)

    def perpendicular_widths(self) -> list[float]:
        """
        Distance between the opposite faces of the cell, per lattice direction.

        For a cell spanned by a, b, c the spacing of the planes normal to the
        (b, c) face is V / |b x c|, and so on.  For an orthogonal box this is
        just prd.  Bins have to be at least a cutoff wide measured this way, not
        along the (longer) lattice vectors.
        """
        xprd, yprd, zprd = (float(v) for v in self.domain.prd)
        xy, xz, yz = (float(v) for v in self.domain.tilt)
        a = np.array([xprd, 0.0, 0.0])
        b = np.array([xy, yprd, 0.0])
        c = np.array([xz, yz, zprd])
        vol = xprd * yprd * zprd
        return [
            vol / np.linalg.norm(np.cross(b, c)),
            vol / np.linalg.norm(np.cross(c, a)),
            vol / np.linalg.norm(np.cross(a, b)),
        ]

    def setup_grid(self, max_cutoff: float) -> None:
        """
        Setup grid dimensions given maximum cutoff distance (2 * r_max + skin).

        Bins are laid out in lamda (fractional) coordinates, where the cell is
        always the unit cube regardless of tilt.  That is what makes wrapping a
        bin index across a periodic boundary a plain modulo: in Cartesian bins,
        stepping one period along z of a tilted box also shifts x by xz, so the
        index arithmetic would be wrong.  LAMMPS avoids the problem differently,
        by binning the Cartesian bounding box and relying on ghost atoms, which
        taichimps does not have.  For an orthogonal box lamda bins and Cartesian
        bins coincide.
        """
        cut = max_cutoff + self.skin
        if cut <= 0:
            cut = 1.0

        widths = self.perpendicular_widths()
        gx = max(1, int(np.floor(widths[0] / cut)))
        gy = max(1, int(np.floor(widths[1] / cut)))
        gz = max(1, int(np.floor(widths[2] / cut)))

        # A periodic dimension binned into exactly 2 cells is degenerate: the
        # -1 and +1 stencil offsets wrap onto the same cell, so every candidate
        # in it would be visited twice and end up in the list twice (doubling
        # the contact force).  Collapse such a dimension to a single bin, which
        # the stencil then visits exactly once.
        if self.domain.periodicity[0] == 1 and gx == 2:
            gx = 1
        if self.domain.periodicity[1] == 1 and gy == 2:
            gy = 1
        if self.domain.periodicity[2] == 1 and gz == 2:
            gz = 1

        total_cells = gx * gy * gz
        if total_cells > self.max_grid_cells:
            # Scale down grid resolution to fit memory
            factor = (self.max_grid_cells / total_cells) ** (1.0 / 3.0)
            gx = max(1, int(gx * factor))
            gy = max(1, int(gy * factor))
            gz = max(1, int(gz * factor))

        self.grid_dim[None] = ti.Vector([gx, gy, gz])
        # Bin size in lamda units; the unit cube is divided into gx*gy*gz bins.
        self.cell_size[None] = ti.Vector(
            [1.0 / gx, 1.0 / gy, 1.0 / gz], dt=self.float_type
        )

    @ti.func
    def get_cell_coord(self, pos):
        gdim = self.grid_dim[None]
        csize = self.cell_size[None]
        # Bin in lamda coordinates. Domain reads the box from device fields, so
        # this follows a deforming box instead of the step-zero one.
        lamda = self.domain.lamda_of(pos)
        cx = ti.cast(ti.floor(lamda[0] / csize[0]), ti.i32)
        cy = ti.cast(ti.floor(lamda[1] / csize[1]), ti.i32)
        cz = ti.cast(ti.floor(lamda[2] / csize[2]), ti.i32)

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
        """
        Bin the particles by counting sort: count, prefix sum, scatter.

        Every step uses an atomic, so the result does not depend on how the
        loops are scheduled.  `cell_particles[cell_start[c] : cell_start[c+1]]`
        is then the membership of bin c, contiguous.
        """
        gdim = self.grid_dim[None]
        gx, gy, gz = gdim[0], gdim[1], gdim[2]
        gxy = gx * gy
        num_cells = gxy * gz

        for c in range(num_cells):
            self.cell_count[c] = 0

        for i in range(nlocal):
            cell = self.get_cell_coord(x[i])
            c_idx = cell[0] + cell[1] * gx + cell[2] * gxy
            self.cell_of[i] = c_idx
            ti.atomic_add(self.cell_count[c_idx], 1)

        # Exclusive prefix sum. Serial, but over bins rather than particles, and
        # kept inside the kernel so no value has to travel to the host.
        ti.loop_config(serialize=True)
        for c in range(num_cells):
            self.cell_start[c + 1] = self.cell_start[c] + self.cell_count[c]

        for c in range(num_cells):
            self.cell_fill[c] = 0

        for i in range(nlocal):
            c_idx = self.cell_of[i]
            slot = ti.atomic_add(self.cell_fill[c_idx], 1)
            self.cell_particles[self.cell_start[c_idx] + slot] = i

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
        per = self.domain.periodicity_f[None]
        px, py, pz = per[0], per[1], per[2]

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

                # A periodic dimension with a single bin is already fully
                # covered by the dx == 0 offset; wrapping the others onto it
                # would visit the same cell three times.
                skip = (
                    (px == 1 and gx == 1 and dx != 0)
                    or (py == 1 and gy == 1 and dy != 0)
                    or (pz == 1 and gz == 1 and dz != 0)
                )

                # Handle boundary wrapping
                if px == 1:
                    nx = (nx % gx + gx) % gx
                if py == 1:
                    ny = (ny % gy + gy) % gy
                if pz == 1:
                    nz = (nz % gz + gz) % gz

                if not skip and 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz:
                    c_idx = nx + ny * gx + nz * gxy
                    for p in range(self.cell_start[c_idx], self.cell_start[c_idx + 1]):
                        j = self.cell_particles[p]
                        if j > i:
                            dpos = self.domain.minimum_image(pos_i - x[j])
                            rsq = dpos.dot(dpos)
                            rad_sum = r_i + radius[j] + self.skin
                            if rsq < rad_sum * rad_sum:
                                # Keep scanning even once full, so that the
                                # overflow is detected rather than hidden by
                                # an early exit from the bin traversal.
                                if count < self.max_neighbors_per_atom:
                                    self.neighbors[i, count] = j
                                    count += 1
                                else:
                                    self.overflow[None] = 1

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
        self.overflow[None] = 0
        self.build_neighbor_list(atom.nlocal, atom.x, atom.radius)
        if self.overflow[None] != 0:
            raise RuntimeError(
                "Neighbor list overflow: more than "
                f"{self.max_neighbors_per_atom} neighbors for at least one particle. "
                "Increase max_neighbors_per_atom, or reduce the neighbor skin."
            )
        self.store_x0(atom.nlocal, atom.x)
        self.has_built = True
        self.build_count += 1
        self.last_build_step = self.current_step
