"""
Contact History Management for granular shear/sliding history.
Reference: LAMMPS src/GRANULAR/pair_gran_hooke_history.cpp, src/fix_neigh_history.cpp
License: GPL v2 compatible / taichimps MIT reimplementation

History is stored per neighbor-list slot (i, k), which means it is tied to the
*ordering* of the neighbor list.  Whenever the neighbor list is rebuilt that
ordering changes, so the history has to be carried over explicitly or it is
silently destroyed.  LAMMPS does this in FixNeighHistory: `pre_exchange()`
saves the currently touching partners keyed by atom *tag*, and `post_neighbor()`
walks the freshly built list and restores the saved values for every partner it
recognises, zeroing the rest.  `save_state()` / `restore_state()` below are the
Taichi equivalent of those two steps.
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem


@ti.data_oriented
class ContactHistory:
    """
    Tracks tangential shear displacement delta_s between contact pairs.
    Dimensions: (max_atoms, max_neighbors_per_atom, 3)

    `partner[i, k]` holds the *tag* of the touching partner for slot (i, k), or
    -1 when the slot is not in contact.  Tags rather than local indices, because
    indices are reshuffled by AtomSystem.filter_particles() (used for trimming a
    specimen) and would silently mis-associate history afterwards.  Tags are
    1-based, so -1 is unambiguous.
    """

    def __init__(
        self,
        max_atoms: int,
        max_neighbors: int = 64,
        float_type: Any = ti.f64,
    ) -> None:
        self.max_atoms = max_atoms
        self.max_neighbors = max_neighbors
        self.float_type = float_type

        # tag of the partner touching in slot (i, k), -1 when not touching
        self.partner = ti.field(dtype=ti.i32, shape=(max_atoms, max_neighbors))
        # shear displacement vector (delta_s_x, delta_s_y, delta_s_z)
        self.shear = ti.Vector.field(
            3, dtype=float_type, shape=(max_atoms, max_neighbors)
        )
        # rolling displacement vector (optional, for models with rolling friction)
        self.rolling = ti.Vector.field(
            3, dtype=float_type, shape=(max_atoms, max_neighbors)
        )

        # Save buffers, keyed by partner tag, used to carry history across a
        # neighbor list rebuild.  Only slots that are actually touching are
        # saved, so `save_count` is the coordination number, not max_neighbors.
        self.save_tag = ti.field(dtype=ti.i32, shape=(max_atoms, max_neighbors))
        self.save_shear = ti.Vector.field(
            3, dtype=float_type, shape=(max_atoms, max_neighbors)
        )
        self.save_rolling = ti.Vector.field(
            3, dtype=float_type, shape=(max_atoms, max_neighbors)
        )
        self.save_count = ti.field(dtype=ti.i32, shape=max_atoms)

        self.reset()

    @ti.kernel
    def reset(self):
        for i in range(self.max_atoms):
            self.save_count[i] = 0
            for k in range(self.max_neighbors):
                self.partner[i, k] = -1
                self.shear[i, k] = ti.Vector([0.0, 0.0, 0.0])
                self.rolling[i, k] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def save_state_kernel(self, nlocal: ti.i32):
        """Compact the touching contacts of every atom into the tag-keyed buffers."""
        for i in range(nlocal):
            m = 0
            for k in range(self.max_neighbors):
                ptag = self.partner[i, k]
                if ptag >= 0:
                    self.save_tag[i, m] = ptag
                    self.save_shear[i, m] = self.shear[i, k]
                    self.save_rolling[i, m] = self.rolling[i, k]
                    m += 1
            self.save_count[i] = m

    @ti.kernel
    def restore_state_kernel(
        self,
        nlocal: ti.i32,
        tag: ti.template(),
        num_neighbors: ti.template(),
        neighbors: ti.template(),
    ):
        """Repopulate every slot of the new neighbor list from the saved buffers."""
        for i in range(nlocal):
            n = num_neighbors[i]
            cnt = self.save_count[i]
            for k in range(self.max_neighbors):
                found = -1
                jtag = -1
                if k < n:
                    jtag = tag[neighbors[i, k]]
                    for m in range(cnt):
                        if self.save_tag[i, m] == jtag:
                            found = m
                            break

                if found >= 0:
                    self.partner[i, k] = jtag
                    self.shear[i, k] = self.save_shear[i, found]
                    self.rolling[i, k] = self.save_rolling[i, found]
                else:
                    # Either an unused slot or a partner that was not touching
                    # before: LAMMPS zeroes these (allflags[jj] = 0).
                    self.partner[i, k] = -1
                    self.shear[i, k] = ti.Vector([0.0, 0.0, 0.0])
                    self.rolling[i, k] = ti.Vector([0.0, 0.0, 0.0])

    def save_state(self, atom: AtomSystem) -> None:
        """Save touching-contact history before the neighbor list is rebuilt."""
        if atom.nlocal == 0:
            return
        self.save_state_kernel(atom.nlocal)

    def restore_state(self, atom: AtomSystem, nlist: Any) -> None:
        """Restore history onto the freshly rebuilt neighbor list."""
        if atom.nlocal == 0:
            return
        self.restore_state_kernel(
            atom.nlocal,
            atom.tag,
            nlist.num_neighbors,
            nlist.neighbors,
        )
