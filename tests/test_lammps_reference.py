"""
Numerical parity against the real LAMMPS binary.

These tests run the same configuration through LAMMPS and through taichimps and
compare forces and torques particle by particle.  They skip when no LAMMPS
executable is available (see tests/lammps_harness.py).

The configurations deliberately include rotating particles in a periodic box:
the rotational term of the tangential relative velocity and the periodic virial
are exactly the places where the two implementations had drifted apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent))

from lammps_harness import (
    force_array,
    lmp_executable,
    run_lammps,
    torque_array,
)

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.GRANULAR.hooke_history import GranHookeHistory
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation

pytestmark = pytest.mark.skipif(
    lmp_executable() is None, reason="no LAMMPS executable available"
)

BOX = 0.02
RADIUS = 0.0025
DENSITY = 2650.0
KN, KT = 1.0e6, 2.0e5
GAMMAN, GAMMAT = 500.0, 250.0
XMU = 0.5
DT = 1.0e-6
SKIN = 0.0005


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def _config(seed: int, n_side: int = 3):
    """A jittered lattice of overlapping spheres with random spins."""
    rng = np.random.default_rng(seed)
    spacing = BOX / n_side
    grid = [
        [(a + 0.5) * spacing, (b + 0.5) * spacing, (c + 0.5) * spacing]
        for a in range(n_side)
        for b in range(n_side)
        for c in range(n_side)
    ]
    x = np.asarray(grid, dtype=np.float64)
    # Push particles together so that neighbours actually overlap.
    x += rng.uniform(-0.05 * spacing, 0.05 * spacing, size=x.shape)
    x = np.mod(x, BOX)
    n = len(x)
    radius = np.full(n, spacing * 0.52)
    density = np.full(n, DENSITY)
    v = rng.normal(scale=0.02, size=x.shape)
    omega = rng.normal(scale=20.0, size=x.shape)
    return x, radius, density, v, omega


def _run_taichimps(x, radius, density, v, omega, steps):
    domain = Domain(
        boxlo=[0.0, 0.0, 0.0], boxhi=[BOX, BOX, BOX], boundary=("p", "p", "p")
    )
    n = len(x)
    atom = AtomSystem(max_atoms=n)
    atom.add_particles(x=x, radius=radius, density=density, v=v, omega=omega)
    neighbor = NeighborList(
        domain=domain, max_atoms=n, max_neighbors_per_atom=64, skin=SKIN
    )
    # Match `neigh_modify delay 0 every 1 check no` in the LAMMPS input.
    neighbor.check = False
    neighbor.every = 1
    history = ContactHistory(max_atoms=n, max_neighbors=64)
    pair = GranHookeHistory(
        domain=domain, kn=KN, gamman=GAMMAN, kt=KT, gammat=GAMMAT, xmu=XMU, dampflag=1
    )
    sim = Simulation(
        domain=domain, atom=atom, neighbor=neighbor, history=history, pair=pair, dt=DT
    )
    sim.add_fix(FixNVESphere(domain=domain))
    if steps == 0:
        sim.init_simulation()
    else:
        sim.run(steps)
    return (
        atom.f.to_numpy()[:n].copy(),
        atom.torque.to_numpy()[:n].copy(),
    )


def _assert_close(got, ref, label):
    scale = np.abs(ref).max()
    assert scale > 0.0, f"reference {label} is all zero; the test case has no contacts"
    err = np.abs(got - ref).max() / scale
    assert err < 1e-9, (
        f"{label} differs from LAMMPS by {err:.3e} (relative to {scale:.3e})\n"
        f"taichimps[0:3]=\n{got[:3]}\nLAMMPS[0:3]=\n{ref[:3]}"
    )


@pytest.mark.parametrize("steps", [0, 1, 25])
def test_hooke_history_matches_lammps(tmp_path, steps):
    """
    Forces and torques must agree with LAMMPS at setup, after one step, and
    after enough steps that the shear history dominates the tangential force.
    """
    x, radius, density, v, omega = _config(seed=3)

    frames = run_lammps(
        tmp_path / f"lmp_{steps}",
        x=x,
        radius=radius,
        density=density,
        v=v,
        omega=omega,
        boxlo=(0.0, 0.0, 0.0),
        boxhi=(BOX, BOX, BOX),
        pair_style=f"gran/hooke/history {KN} {KT} {GAMMAN} {GAMMAT} {XMU} 1",
        steps=steps,
        dt=DT,
        skin=SKIN,
    )
    ref = frames[-1]
    f_ref, tq_ref = force_array(ref), torque_array(ref)

    f_got, tq_got = _run_taichimps(x, radius, density, v, omega, steps)

    _assert_close(f_got, f_ref, "force")
    _assert_close(tq_got, tq_ref, "torque")
