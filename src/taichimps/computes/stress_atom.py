"""
ComputeStressAtom matching LAMMPS compute stress/atom.
Reference: LAMMPS src/compute_stress_atom.cpp

LAMMPS defines the per-atom stress as the negated per-atom virial, in units of
pressure*volume (it is NOT divided by a per-atom volume).  The virial itself is
tallied per contact by the pair styles into AtomSystem.virial, so this compute
only has to negate it and optionally add the kinetic term.
"""

from typing import Any

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain


@ti.data_oriented
class ComputeStressAtom:
    """Per-atom stress tensor (6 components, [xx, yy, zz, xy, xz, yz])."""

    def __init__(self, max_atoms: int, float_type: Any = ti.f64) -> None:
        self.float_type = float_type
        self.max_atoms = max_atoms
        self.stress = ti.Vector.field(6, dtype=float_type, shape=max_atoms)
        self.macro_stress = ti.field(dtype=float_type, shape=6)

    @ti.kernel
    def compute_stress_kernel(
        self,
        nlocal: ti.i32,
        atom: ti.template(),
        keflag: ti.i32,
    ):
        for k in ti.static(range(6)):
            self.macro_stress[k] = 0.0

        for i in range(nlocal):
            s = -atom.virial[i]
            if keflag != 0:
                m = atom.rmass[i]
                s[0] -= m * atom.v[i][0] * atom.v[i][0]
                s[1] -= m * atom.v[i][1] * atom.v[i][1]
                s[2] -= m * atom.v[i][2] * atom.v[i][2]
                s[3] -= m * atom.v[i][0] * atom.v[i][1]
                s[4] -= m * atom.v[i][0] * atom.v[i][2]
                s[5] -= m * atom.v[i][1] * atom.v[i][2]
            self.stress[i] = s
            for k in ti.static(range(6)):
                self.macro_stress[k] += s[k]

    def compute(
        self,
        atom: AtomSystem,
        domain: Domain,
        kinetic: bool = False,
    ) -> np.ndarray:
        """
        Per-atom stress as an (nlocal, 6) array in pressure*volume units.

        `kinetic=False` matches `compute stress/atom NULL pair`, where no
        temperature compute is supplied so only the pair virial contributes.
        """
        if atom.nlocal == 0:
            return np.zeros((0, 6), dtype=np.float64)
        self.compute_stress_kernel(
            atom.nlocal,
            atom,
            1 if kinetic else 0,
        )
        return self.stress.to_numpy()[: atom.nlocal]

    def compute_macro(
        self,
        atom: AtomSystem,
        domain: Domain,
        kinetic: bool = False,
    ) -> np.ndarray:
        """
        Volume-averaged Cauchy stress, i.e. sum of the per-atom stress divided
        by the box volume.  This is the negative of the pressure tensor, so
        compression comes out negative here and positive in Computes.
        """
        if atom.nlocal == 0:
            return np.zeros(6, dtype=np.float64)
        vol = domain.volume
        if vol <= 0.0:
            return np.zeros(6, dtype=np.float64)
        self.compute(atom, domain, kinetic=kinetic)
        return self.macro_stress.to_numpy() / vol
