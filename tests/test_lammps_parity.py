"""
Regression tests for behaviours that must match LAMMPS but silently did not.

Each test here pins down one specific discrepancy that was found by reading the
LAMMPS sources in the sibling checkout, and each one fails on the code as it
stood before the corresponding fix:

  * the pairwise virial must be translation invariant under PBC
  * contact shear history must survive a neighbor list rebuild
  * the rotational part of the tangential relative velocity must have the sign
    LAMMPS uses, so that friction damps spin rather than driving it
"""

import numpy as np
import pytest
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.computes.thermo import Computes
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.GRANULAR.hooke_history import GranHookeHistory
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def _make_sim(positions, radius=0.5, box=6.0, skin=0.1, xmu=0.5, velocities=None,
              omegas=None, dt=1e-4):
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[box, box, box], boundary=("p", "p", "p"))
    n = len(positions)
    atom = AtomSystem(max_atoms=max(n, 8))
    atom.add_particles(
        x=np.asarray(positions, dtype=np.float64),
        radius=radius,
        density=2650.0,
        v=velocities,
        omega=omegas,
    )
    neighbor = NeighborList(domain=domain, max_atoms=max(n, 8),
                            max_neighbors_per_atom=32, skin=skin)
    history = ContactHistory(max_atoms=max(n, 8), max_neighbors=32)
    pair = GranHookeHistory(domain=domain, kn=1.0e5, gamman=10.0, kt=5.0e4,
                            gammat=5.0, xmu=xmu)
    sim = Simulation(domain=domain, atom=atom, neighbor=neighbor, history=history,
                     pair=pair, dt=dt)
    sim.add_fix(FixNVESphere(domain=domain))
    return sim


def _random_packing(rng, n=60, box=6.0, radius=0.5):
    """Positions on a jittered lattice, dense enough that particles overlap."""
    side = int(np.ceil(n ** (1.0 / 3.0)))
    spacing = box / side
    pts = []
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if len(pts) >= n:
                    break
                pts.append([(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing])
    pts = np.asarray(pts[:n], dtype=np.float64)
    pts += rng.uniform(-0.02 * spacing, 0.02 * spacing, size=pts.shape)
    return pts % box


def test_pressure_tensor_is_translation_invariant():
    """
    The virial must not depend on where the box origin sits.

    LAMMPS tallies del_ij x f_ij per contact; summing x_i . f_i instead (which
    is what taichimps used to do) only agrees for an isolated cluster, and
    drifts with the origin as soon as contacts straddle a periodic boundary.
    """
    rng = np.random.default_rng(7)
    box = 6.0
    radius = 0.52 * (box / 4)
    pts = _random_packing(rng, n=64, box=box, radius=radius)

    computes = Computes()

    def pressure_for(offset):
        sim = _make_sim((pts + offset) % box, radius=radius, box=box)
        sim.init_simulation()
        return computes.compute_pressure_tensor(sim.atom, sim.domain, kinetic=False)

    p_ref = pressure_for(np.zeros(3))
    p_shift = pressure_for(np.array([0.37 * box, 0.11 * box, 0.53 * box]))

    assert np.linalg.norm(p_ref) > 0.0, "test packing produced no contacts"
    np.testing.assert_allclose(p_shift, p_ref, rtol=1e-9, atol=1e-6 * np.linalg.norm(p_ref))


def test_shear_history_survives_neighbor_rebuild():
    """
    Rebuilding the neighbor list must not reset the tangential history.

    Two settings that differ only in how often the list is rebuilt have to give
    the same trajectory. Before ContactHistory.save_state/restore_state existed,
    every rebuild silently zeroed the shear of every contact.
    """
    rng = np.random.default_rng(11)
    box = 6.0
    radius = 0.52 * (box / 4)
    pts = _random_packing(rng, n=64, box=box, radius=radius)
    vel = rng.normal(scale=0.05, size=pts.shape)
    omg = rng.normal(scale=0.5, size=pts.shape)

    def run(rebuild_every_step):
        sim = _make_sim(pts, radius=radius, box=box, velocities=vel, omegas=omg)
        if rebuild_every_step:
            # LAMMPS: neigh_modify every 1 delay 0 check no
            sim.neighbor.check = False
            sim.neighbor.every = 1
        else:
            sim.neighbor.check = True
            sim.neighbor.every = 10**6
        sim.run(40)
        return sim.atom.f.to_numpy()[: sim.atom.nlocal].copy()

    f_rebuilt = run(True)
    f_kept = run(False)

    scale = np.abs(f_kept).max()
    assert scale > 0.0, "test packing produced no contacts"
    np.testing.assert_allclose(f_rebuilt, f_kept, rtol=1e-8, atol=1e-8 * scale)


def test_friction_damps_spin_rather_than_driving_it():
    """
    Sign check on the rotational term of the tangential relative velocity.

    LAMMPS computes vtr = vt - W x n with W = r_i*omega_i + r_j*omega_j. For a
    pair whose only motion is a spin of particle i about z, the contact point
    on i moves along +y, so friction must push it along -y and apply a torque
    that opposes the spin. The previous sign gave the opposite, which fed
    energy into the rotation instead of dissipating it.
    """
    radius = 0.5
    overlap = 0.01
    d = 2.0 * radius - overlap
    spin = 3.0

    sim = _make_sim(
        [[3.0, 3.0, 3.0], [3.0 + d, 3.0, 3.0]],
        radius=radius,
        omegas=np.array([[0.0, 0.0, spin], [0.0, 0.0, 0.0]]),
    )
    sim.init_simulation()
    sim.step()

    f = sim.atom.f.to_numpy()[:2]
    torque = sim.atom.torque.to_numpy()[:2]

    # Surface of particle 0 at the contact moves along +y, so friction acts along -y.
    assert f[0][1] < 0.0
    # ... and the resulting torque opposes the positive-z spin.
    assert torque[0][2] < 0.0
    # Newton's third law on the pair force.
    np.testing.assert_allclose(f[0] + f[1], np.zeros(3), atol=1e-12 * np.abs(f).max())
