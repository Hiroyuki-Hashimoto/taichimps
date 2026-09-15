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

import numpy as np
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

        # History is stored per *pair*, indexed by position in the neighbour
        # list's flat pair array, not per (atom, slot). The force kernel runs
        # one thread per pair, so this puts a pair's history where the thread
        # that needs it already is; the per-atom form made every thread gather
        # from a row of its own. GeoTaichi holds it the same way, inside the
        # contact record (src/dem/structs/BaseStruct.py, ContactTable).
        self.max_pairs = max_atoms * max_neighbors
        # tag of the partner touching in pair nc, -1 when not touching
        self.partner = ti.field(dtype=ti.i32, shape=self.max_pairs)
        # shear displacement vector (delta_s_x, delta_s_y, delta_s_z)
        self.shear = ti.Vector.field(3, dtype=float_type, shape=self.max_pairs)
        # rolling displacement vector (optional, for models with rolling friction)
        self.rolling = ti.Vector.field(3, dtype=float_type, shape=self.max_pairs)

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

        # Set while the save buffers hold history that has not yet been placed
        # on a neighbor list -- currently only right after read_restart.  It
        # suppresses the next save_state(), mirroring the LAMMPS note on
        # FixNeighHistory::pre_exchange ("do not call during setup of run ...
        # because there is no guarantee of a current NDS").
        self.setup_pending = False

        self.reset()

    @ti.kernel
    def reset(self):
        for i in range(self.max_atoms):
            self.save_count[i] = 0
        for nc in range(self.max_pairs):
            self.partner[nc] = -1
            self.shear[nc] = ti.Vector([0.0, 0.0, 0.0])
            self.rolling[nc] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def save_state_kernel(self, nlocal: ti.i32, npairs: ti.template(),
                          pair_i: ti.template()):
        """
        Compact the touching contacts of every atom into the tag-keyed buffers.

        The order within an atom's buffer is whatever the atomics hand out, and
        that is fine: restore_state_kernel looks a partner up by tag, never by
        position.
        """
        for i in range(nlocal):
            self.save_count[i] = 0
        for nc in range(npairs[None]):
            ptag = self.partner[nc]
            if ptag >= 0:
                i = pair_i[nc]
                m = ti.atomic_add(self.save_count[i], 1)
                self.save_tag[i, m] = ptag
                self.save_shear[i, m] = self.shear[nc]
                self.save_rolling[i, m] = self.rolling[nc]

    @ti.kernel
    def restore_state_kernel(
        self,
        tag: ti.template(),
        npairs: ti.template(),
        pair_i: ti.template(),
        pair_j: ti.template(),
    ):
        """Repopulate every pair of the new neighbor list from the saved buffers."""
        for nc in range(npairs[None]):
            i = pair_i[nc]
            jtag = tag[pair_j[nc]]
            found = -1
            for m in range(self.save_count[i]):
                if self.save_tag[i, m] == jtag:
                    found = m
                    break

            if found >= 0:
                self.partner[nc] = jtag
                self.shear[nc] = self.save_shear[i, found]
                self.rolling[nc] = self.save_rolling[i, found]
            else:
                # A partner that was not touching before: LAMMPS zeroes these
                # (allflags[jj] = 0).
                self.partner[nc] = -1
                self.shear[nc] = ti.Vector([0.0, 0.0, 0.0])
                self.rolling[nc] = ti.Vector([0.0, 0.0, 0.0])

    def save_state(self, atom: AtomSystem, nlist: Any) -> None:
        """Save touching-contact history before the neighbor list is rebuilt."""
        if atom.nlocal == 0 or self.setup_pending:
            return
        self.save_state_kernel(atom.nlocal, nlist.npairs, nlist.pair_i)

    def restore_state(self, atom: AtomSystem, nlist: Any) -> None:
        """Restore history onto the freshly rebuilt neighbor list."""
        if atom.nlocal == 0:
            return
        self.restore_state_kernel(
            atom.tag, nlist.npairs, nlist.pair_i, nlist.pair_j
        )
        self.setup_pending = False

    def touching_by_atom(
        self, nlist: Any, nlocal: int
    ) -> list[list[tuple[int, np.ndarray]]]:
        """
        The touching contacts of each local atom, as (partner tag, shear).

        History lives per pair, so recovering "what is atom i touching" means
        walking the pair list. Only write_restart needs this, once per file.
        """
        npairs = int(nlist.npairs[None])
        pair_i = nlist.pair_i.to_numpy()[:npairs]
        partner = self.partner.to_numpy()[:npairs]
        shear = self.shear.to_numpy()[:npairs]
        out: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(nlocal)]
        for nc in np.nonzero(partner >= 0)[0]:
            i = int(pair_i[nc])
            if i < nlocal:
                out[i].append((int(partner[nc]), shear[nc]))
        return out

    def load_restart(
        self,
        partner_tags: list[np.ndarray],
        values: list[np.ndarray],
    ) -> None:
        """
        Seed the save buffers from a restart file, keyed by partner tag.

        This is the counterpart of FixNeighHistory::unpack_restart: the values
        land in the neighbor-data structures, and the first neighbor build of
        the run (setup_post_neighbor) puts them onto the list.  Entry `k` of
        the lists belongs to local atom `k`, i.e. the file order in which the
        atoms were handed to AtomSystem.add_particles().

        LAMMPS stores each contact on both partners, the second copy negated
        (FixNeighHistory::pre_exchange_no_newton).  An entry found under atom
        `i` for partner `j` is therefore always in the i-to-j convention, which
        is the one the pair kernels use for slot (i, j).
        """
        n = min(len(partner_tags), self.max_atoms)
        tags = np.full((self.max_atoms, self.max_neighbors), -1, dtype=np.int32)
        shear = np.zeros((self.max_atoms, self.max_neighbors, 3), dtype=np.float64)
        count = np.zeros(self.max_atoms, dtype=np.int32)
        for i in range(n):
            t = np.asarray(partner_tags[i], dtype=np.int32)
            v = np.asarray(values[i], dtype=np.float64)
            m = len(t)
            if m > self.max_neighbors:
                raise ValueError(
                    f"atom {i} has {m} stored contacts but the history only "
                    f"has {self.max_neighbors} slots per atom"
                )
            tags[i, :m] = t
            shear[i, :m, : v.shape[1]] = v[:, :3]
            count[i] = m
        self.save_tag.from_numpy(tags)
        self.save_shear.from_numpy(shear)
        self.save_count.from_numpy(count)
        self.setup_pending = True
