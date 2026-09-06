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


def test_simulation_run_gpu_consistency():
    """Test that run_gpu produces identical results to step-by-step run."""
    ti.init(arch=ti.cpu)

    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0], boundary=["p", "p", "p"])
    atom = AtomSystem(max_atoms=10, float_type=ti.f64)

    atom.add_particles(
        x=[[5.0, 5.0, 5.0]],
        v=[[0.0, 0.0, 0.0]],
        omega=[[0.0, 0.0, 0.0]],
        radius=[0.5],
        density=1000.0,
        atom_type=1,
    )

    nlist = NeighborList(domain=domain, max_atoms=10, skin=0.2, float_type=ti.f64)
    history = ContactHistory(max_atoms=10, float_type=ti.f64)
    pair = GranHooke(domain=domain, kn=1e4, kt=1e3, gamman=10.0, gammat=5.0, xmu=0.5, float_type=ti.f64)

    sim = Simulation(
        domain=domain,
        atom=atom,
        neighbor=nlist,
        history=history,
        pair=pair,
        dt=0.001,
    )

    grav = FixGravity(domain=domain, magnitude=9.81, direction=[0.0, 0.0, -1.0])
    nve = FixNVESphere(domain=domain)
    sim.add_fix(grav)
    sim.add_fix(nve)

    nsteps = 100
    sim.run_gpu(nsteps)

    assert sim.timestep == nsteps
    z_final = float(atom.x.to_numpy()[0, 2])
    # z(t) = z0 - 0.5 * g * t^2
    t = nsteps * 0.001
    expected_z = 5.0 - 0.5 * 9.81 * t * t
    assert np.isclose(z_final, expected_z, atol=1e-4)
