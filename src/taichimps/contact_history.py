"""
Contact History Management for granular shear/sliding history.
Reference: LAMMPS src/GRANULAR/pair_gran_hooke_history.cpp, FixContactHistory
License: GPL v2 compatible / taichimps MIT reimplementation
"""

from typing import Any

import taichi as ti


@ti.data_oriented
class ContactHistory:
    """
    Tracks tangential shear displacement delta_s between contact pairs.
    Dimensions: (max_atoms, max_neighbors_per_atom, 3)
    Also stores partner tag/id to preserve history across neighbor rebuilds.
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

        # partner atom index j for slot (i, k)
        self.partner = ti.field(dtype=ti.i32, shape=(max_atoms, max_neighbors))
        # shear displacement vector (delta_s_x, delta_s_y, delta_s_z)
        self.shear = ti.Vector.field(
            3, dtype=float_type, shape=(max_atoms, max_neighbors)
        )
        # rolling displacement vector (optional, for models with rolling friction)
        self.rolling = ti.Vector.field(
            3, dtype=float_type, shape=(max_atoms, max_neighbors)
        )

        self.reset()

    @ti.kernel
    def reset(self):
        for i in range(self.max_atoms):
            for k in range(self.max_neighbors):
                self.partner[i, k] = -1
                self.shear[i, k] = ti.Vector([0.0, 0.0, 0.0])
                self.rolling[i, k] = ti.Vector([0.0, 0.0, 0.0])

    def compress_and_update(self, nlist: Any) -> None:
        """Update contact history to align with reconstructed neighbor list."""
