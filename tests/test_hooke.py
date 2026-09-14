"""
Unit tests for Hooke contact force model.
Tests normal spring-dashpot contact between two spheres.
"""

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.GRANULAR.hooke import GranHooke
from taichimps.neighbor import NeighborList


def test_hooke_normal_contact():
    ti.init(arch=ti.cpu)
    domain = Domain(boxlo=[-10.0, -10.0, -10.0], boxhi=[10.0, 10.0, 10.0], boundary=["f", "f", "f"])
    atom = AtomSystem(max_atoms=10)

    # Particle 0 and 1 with radius 1.0, distance = 1.9 (overlap delta = 0.1)
    # Moving towards each other: v0 = [0.1, 0, 0], v1 = [-0.1, 0, 0]
    atom.add_particles(
        tag=[1, 2],
        atom_type=[1, 1],
        x=[[0.0, 0.0, 0.0], [1.9, 0.0, 0.0]],
        v=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        radius=[1.0, 1.0],
        density=[1000.0, 1000.0],
    )

    nlist = NeighborList(domain=domain, skin=0.2, max_atoms=10, max_neighbors=10)
    nlist.build(atom)
    history = ContactHistory(max_atoms=10, max_neighbors=10)

    # kn = 1000.0, gamman = 10.0, kt = 0.0, gammat = 0.0, xmu = 0.0, dampflag = 0
    pair = GranHooke(domain=domain, kn=1000.0, gamman=10.0, kt=0.0, gammat=0.0, xmu=0.0, dampflag=0)

    # Normal damping in LAMMPS is always scaled by the effective mass; dampflag
    # only controls the *tangential* damping term.  So with
    #   m     = 4/3 pi r^3 rho          = 4188.7902...
    #   meff  = m * m / (m + m) = m / 2 = 2094.3951...
    #   dpos  = x0 - x1 = (-1.9, 0, 0),  vnnr = (v0 - v1) . dpos = -0.38
    #   damp  = meff * gamman * vnnr / rsq          = -2204.6212...
    #   ccel  = kn * delta / r - damp               =  2257.2528...
    #   f0    = dpos * ccel                         = (-4288.7902..., 0, 0)
    # Cross-checked against the LAMMPS binary (pair_style gran/hooke,
    # run 0) which reports fx = -4288.79020479 for particle 1.
    expected = 4288.790204786391

    pair.compute(atom, nlist, history, dt=0.001)

    f = atom.f.to_numpy()
    np.testing.assert_allclose(f[0], [-expected, 0.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(f[1], [expected, 0.0, 0.0], rtol=1e-12)

