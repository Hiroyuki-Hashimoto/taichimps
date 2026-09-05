import numpy as np
import pytest
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.neighbor import NeighborList


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    ti.init(arch=ti.cpu)


def test_domain_periodicity():
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0], boundary=["p", "p", "p"])
    atom = AtomSystem(max_atoms=10)
    atom.add_particles(
        tag=[1],
        atom_type=[1],
        x=[[10.5, -0.2, 5.0]],
        radius=[0.5],
        density=[1000.0],
    )
    domain.pbc(atom)

    x_np = atom.x.to_numpy()[0]
    assert 0.0 <= x_np[0] < 10.0
    assert 0.0 <= x_np[1] < 10.0
    assert np.isclose(x_np[0], 0.5)
    assert np.isclose(x_np[1], 9.8)


def test_neighbor_list():
    domain = Domain(boxlo=[-10.0, -10.0, -10.0], boxhi=[10.0, 10.0, 10.0], boundary=["f", "f", "f"])
    atom = AtomSystem(max_atoms=10)
    # Two touching/overlapping particles + one far particle
    atom.add_particles(
        tag=[1, 2, 3],
        atom_type=[1, 1, 1],
        x=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
        radius=[1.0, 1.0, 1.0],
        density=[1000.0, 1000.0, 1000.0],
    )

    nlist = NeighborList(domain=domain, skin=0.2, max_atoms=10, max_neighbors=10)
    nlist.build(atom)

    num_neigh = nlist.num_neighbors.to_numpy()
    neighbors = nlist.neighbors.to_numpy()

    # Particle 0 should have particle 1 as neighbor (since distance 1.5 < 1.0 + 1.0 + 0.2 = 2.2)
    # But Particle 0 should not have particle 2 (5.0 > 2.2)
    assert num_neigh[0] == 1
    assert neighbors[0, 0] == 1


def test_neighbor_skin_displacement_check():
    domain = Domain(boxlo=[-10.0, -10.0, -10.0], boxhi=[10.0, 10.0, 10.0], boundary=["f", "f", "f"])
    atom = AtomSystem(max_atoms=10)
    atom.add_particles(
        tag=[1, 2, 3],
        atom_type=[1, 1, 1],
        x=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
        radius=[1.0, 1.0, 1.0],
        density=[1000.0, 1000.0, 1000.0],
    )

    skin = 0.2
    nlist = NeighborList(domain=domain, skin=skin, max_atoms=10, max_neighbors=10)

    # First call: must build
    rebuilt = nlist.check_and_build(atom, timestep=0)
    assert rebuilt is True
    assert nlist.build_count == 1

    # Second call at timestep 1 without movement: should skip build
    rebuilt = nlist.check_and_build(atom, timestep=1)
    assert rebuilt is False
    assert nlist.build_count == 1

    # Move particle within 0.5 * skin (e.g. 0.08 < 0.1): should skip build
    atom.x[0] = ti.Vector([0.08, 0.0, 0.0])
    rebuilt = nlist.check_and_build(atom, timestep=2)
    assert rebuilt is False
    assert nlist.build_count == 1

    # Move particle beyond 0.5 * skin (0.12 > 0.1): should rebuild
    atom.x[0] = ti.Vector([0.12, 0.0, 0.0])
    rebuilt = nlist.check_and_build(atom, timestep=3)
    assert rebuilt is True
    assert nlist.build_count == 2
