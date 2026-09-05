import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.wall_gran import FixWallGran
from taichimps.GRANULAR.hertz import GranHertz
from taichimps.GRANULAR.hertz_history import GranHertzHistory
from taichimps.GRANULAR.hooke_history import GranHookeHistory
from taichimps.GRANULAR.modular import GranularModular
from taichimps.neighbor import NeighborList


def test_hooke_history_tangential():
    ti.init(arch=ti.cpu)
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0], boundary=['p', 'p', 'p'])
    atom = AtomSystem(max_atoms=10)
    atom.add_particles(
        tag=[1, 2],
        atom_type=[1, 1],
        x=[[1.0, 0.0, 0.0], [2.9, 0.0, 0.0]],
        v=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        radius=[1.0, 1.0],
        density=[1000.0, 1000.0],
    )
    nlist = NeighborList(domain=domain, max_atoms=10, max_neighbors=5, skin=0.2)
    history = ContactHistory(max_atoms=10, max_neighbors=5)
    nlist.build(atom)

    pair = GranHookeHistory(domain=domain, kn=1000.0, gamman=0.0, kt=500.0, gammat=0.0, xmu=0.5)
    dt = 0.001
    pair.compute(atom=atom, nlist=nlist, history=history, dt=dt)

    f = atom.f.to_numpy()
    assert f[0, 1] < 0.0
    assert f[1, 1] > 0.0
    assert np.isclose(f[0, 1], -f[1, 1])


def test_hertz_and_history():
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0])
    atom = AtomSystem(max_atoms=10)
    atom.add_particles(
        tag=[1, 2],
        atom_type=[1, 1],
        x=[[1.0, 0.0, 0.0], [2.9, 0.0, 0.0]],
        v=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        radius=[1.0, 1.0],
        density=[1000.0, 1000.0],
    )
    nlist = NeighborList(domain=domain, max_atoms=10, max_neighbors=5, skin=0.2)
    history = ContactHistory(max_atoms=10, max_neighbors=5)
    nlist.build(atom)

    pair_hertz = GranHertz(domain=domain, kn=1000.0, gamman=0.0, kt=0.0, gammat=0.0, xmu=0.0)
    pair_hertz.compute(atom=atom, nlist=nlist, history=history, dt=0.001)

    f = atom.f.to_numpy()
    assert f[0, 0] < 0.0
    assert f[1, 0] > 0.0

    pair_hh = GranHertzHistory(domain=domain, kn=1000.0, gamman=0.0, kt=500.0, gammat=0.0, xmu=0.5)
    atom.f.fill(0.0)
    pair_hh.compute(atom=atom, nlist=nlist, history=history, dt=0.001)
    f2 = atom.f.to_numpy()
    assert f2[0, 0] < 0.0


def test_modular_and_wall():
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0])
    atom = AtomSystem(max_atoms=10)
    atom.add_particles(
        tag=[1],
        atom_type=[1],
        x=[[0.05, 0.5, 0.5]],
        v=[[-1.0, 0.0, 0.0]],
        radius=[0.1],
        density=[1000.0],
    )
    nlist = NeighborList(domain=domain, max_atoms=10, max_neighbors=5, skin=0.1)
    history = ContactHistory(max_atoms=10, max_neighbors=5)

    wall = FixWallGran(domain=domain, wall_axis=0, wall_side=-1, wall_coord=0.0, kn=1e4, gamman=0.0, kt=0.0, gammat=0.0, xmu=0.0)
    wall.post_force(atom=atom, dt=0.001)

    f = atom.f.to_numpy()
    assert f[0, 0] > 0.0

    atom.f.fill(0.0)
    mod = GranularModular(domain=domain, kn=1000.0, gamman=0.0, kt=0.0, gammat=0.0, xmu=0.0)
    mod.compute(atom=atom, nlist=nlist, history=history, dt=0.001)
