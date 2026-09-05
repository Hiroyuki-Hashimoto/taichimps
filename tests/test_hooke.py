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

    # delta = 2.0 - 1.9 = 0.1
    # v_rel_n = (v0 - v1).n = (0.1 - (-0.1)) = 0.2 (approach velocity, distance decreasing)
    # fn = kn * delta - gamman * vn
    # Here vn = (vi - vj).n = -0.2 (negative because particles are approaching)
    # fn = 1000.0 * 0.1 - 10.0 * (-0.2) = 100.0 + 2.0 = 102.0
    # force on 0: f0 = fn * n = 102.0 * (-1, 0, 0) = (-102.0, 0, 0)
    # force on 1: f1 = -fn * n = (+102.0, 0, 0)

    pair.compute(atom, nlist, history, dt=0.001)

    f = atom.f.to_numpy()
    np.testing.assert_allclose(f[0], [-102.0, 0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(f[1], [102.0, 0.0, 0.0], rtol=1e-5)

