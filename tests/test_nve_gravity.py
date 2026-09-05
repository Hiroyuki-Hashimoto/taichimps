import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.gravity import FixGravity
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.GRANULAR.hooke import GranHooke
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation


def test_free_fall_gravity():
    ti.init(arch=ti.cpu)
    domain = Domain(boxlo=[-10.0, -10.0, -10.0], boxhi=[10.0, 10.0, 10.0], boundary=["f", "f", "f"])
    atom = AtomSystem(max_atoms=10)

    # 1 particle at origin at rest
    atom.add_particles(
        tag=[1],
        atom_type=[1],
        x=[[0.0, 0.0, 0.0]],
        v=[[0.0, 0.0, 0.0]],
        radius=[0.1],
        density=[1000.0],
    )

    nlist = NeighborList(domain=domain, skin=0.1, max_atoms=10, max_neighbors=10)
    history = ContactHistory(max_atoms=10, max_neighbors=10)
    pair = GranHooke(domain=domain, kn=100.0, gamman=0.0, kt=0.0, gammat=0.0, xmu=0.0)

    sim = Simulation(domain=domain, atom=atom, neighbor=nlist, history=history, pair=pair, dt=0.01)
    sim.add_fix(FixNVESphere(domain=domain))
    g_acc = 9.81
    sim.add_fix(FixGravity(domain=domain, magnitude=g_acc, direction=[0.0, 0.0, -1.0]))

    # Run for 100 steps -> t = 1.0 s
    nsteps = 100
    sim.run(nsteps)

    # Theoretical: z = -0.5 * g * t^2 = -0.5 * 9.81 * 1.0 = -4.905
    # vz = -g * t = -9.81
    pos = atom.x.to_numpy()[0]
    vel = atom.v.to_numpy()[0]

    np.testing.assert_allclose(vel[2], -g_acc * 1.0, rtol=1e-3)
    np.testing.assert_allclose(pos[2], -0.5 * g_acc * 1.0**2, rtol=1e-3)
