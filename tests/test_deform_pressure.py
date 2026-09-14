"""
fix deform/pressure, checked against the LAMMPS binary.

The rewritten fix has to reproduce three things that the previous version could
not express at all: per-axis pressure targets, mixing a strain-controlled axis
with pressure-controlled ones (the standard triaxial setup), and an affine
remap that keeps the particles positioned correctly relative to the moving box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent))

from lammps_harness import lmp_executable, parse_box, run_lammps

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.deform_pressure import FixDeformPressure
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.GRANULAR.hooke_history import GranHookeHistory
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation

BOX = 0.02
DENSITY = 2650.0
KN, KT = 1.0e6, 2.0e5
GAMMAN, GAMMAT = 500.0, 250.0
XMU = 0.5
DT = 1.0e-6
SKIN = 0.0005
PTARGET = 1.0e4
PGAIN = 1.0e-5
# High enough that the proportional response, not the clamp, sets the rate.
MAXRATE = 100.0
STEPS = 20


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def _config(seed: int = 13, n_side: int = 3):
    rng = np.random.default_rng(seed)
    spacing = BOX / n_side
    x = np.asarray(
        [
            [(a + 0.5) * spacing, (b + 0.5) * spacing, (c + 0.5) * spacing]
            for a in range(n_side)
            for b in range(n_side)
            for c in range(n_side)
        ],
        dtype=np.float64,
    )
    x += rng.uniform(-0.05 * spacing, 0.05 * spacing, size=x.shape)
    x = np.mod(x, BOX)
    n = len(x)
    return (
        x,
        np.full(n, spacing * 0.52),
        np.full(n, DENSITY),
        rng.normal(scale=0.02, size=x.shape),
        rng.normal(scale=20.0, size=x.shape),
    )


def _run_taichimps(x, radius, density, v, omega, axes, couple, steps):
    domain = Domain(
        boxlo=[0.0, 0.0, 0.0], boxhi=[BOX, BOX, BOX], boundary=("p", "p", "p")
    )
    n = len(x)
    atom = AtomSystem(max_atoms=n)
    atom.add_particles(x=x, radius=radius, density=density, v=v, omega=omega)
    neighbor = NeighborList(
        domain=domain, max_atoms=n, max_neighbors_per_atom=64, skin=SKIN
    )
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
    sim.add_fix(
        FixDeformPressure(
            domain=domain, axes=axes, couple=couple, max_rate=MAXRATE, nevery=1
        )
    )
    sim.run(steps)
    return np.array([float(domain.prd[d]) for d in range(3)])


def _lammps_box(tmp_path, cfg, deform_line, steps):
    x, radius, density, v, omega = cfg
    run_lammps(
        tmp_path,
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
        extra_fixes=deform_line,
    )
    box = parse_box(tmp_path / "dump.out")[-1]
    return box[:, 1] - box[:, 0]


@pytest.mark.skipif(lmp_executable() is None, reason="no LAMMPS executable available")
def test_per_axis_pressure_matches_lammps(tmp_path):
    """Three independently servo-controlled axes, couple none."""
    cfg = _config()
    deform = (
        f"fix 2 all deform/pressure 1 "
        f"x pressure {PTARGET} {PGAIN} "
        f"y pressure {PTARGET} {PGAIN} "
        f"z pressure {PTARGET} {PGAIN} "
        f"max/rate {MAXRATE}"
    )
    prd_ref = _lammps_box(tmp_path / "lmp", cfg, deform, STEPS)

    axes = {
        d: {"style": "pressure", "ptarget": PTARGET, "pgain": PGAIN} for d in "xyz"
    }
    prd_got = _run_taichimps(*cfg, axes=axes, couple="none", steps=STEPS)

    rel_change = np.abs(prd_ref / BOX - 1.0)
    assert rel_change.min() > 1e-4, (
        f"the box barely moved ({rel_change}); the test would pass trivially"
    )
    # With couple none the three axes follow their own pressure component, so
    # they must end up at different lengths. The previous implementation could
    # only ever drive all three identically.
    assert prd_ref.std() > 0.0
    np.testing.assert_allclose(prd_got, prd_ref, rtol=1e-9)


@pytest.mark.skipif(lmp_executable() is None, reason="no LAMMPS executable available")
def test_triaxial_strain_plus_pressure_matches_lammps(tmp_path):
    """
    Axial true-strain-rate loading with pressure-servoed side walls.

    This is the combination the old implementation could not express at all:
    it only had a single isotropic target applied to all three axes.
    """
    cfg = _config(seed=21)
    rate = -2.0e3  # 1/s; with dt = 1e-6 that is -0.2% strain over 20 steps
    deform = (
        f"fix 2 all deform/pressure 1 "
        f"x pressure {PTARGET} {PGAIN} "
        f"y pressure {PTARGET} {PGAIN} "
        f"z trate {rate} "
        f"max/rate {MAXRATE}"
    )
    prd_ref = _lammps_box(tmp_path / "lmp", cfg, deform, STEPS)

    axes = {
        "x": {"style": "pressure", "ptarget": PTARGET, "pgain": PGAIN},
        "y": {"style": "pressure", "ptarget": PTARGET, "pgain": PGAIN},
        "z": {"style": "trate", "rate": rate},
    }
    prd_got = _run_taichimps(*cfg, axes=axes, couple="none", steps=STEPS)

    # Axial strain must be the prescribed exp(rate * t) ...
    np.testing.assert_allclose(
        prd_ref[2] / BOX, np.exp(rate * STEPS * DT), rtol=1e-9
    )
    # ... and the servoed side walls must have moved differently from it.
    assert abs(prd_ref[0] / BOX - 1.0) > 1e-4
    assert abs(prd_ref[0] - prd_ref[2]) > 1e-6 * BOX
    np.testing.assert_allclose(prd_got, prd_ref, rtol=1e-9)


def test_remap_keeps_particles_placed_relative_to_the_box():
    """
    The affine remap must anchor on the NEW lower bound.

    Anchoring on the old one (which is what this used to do) shifts every
    particle by half the box-length change on every update, so a particle that
    starts at the centre of the box drifts away from it.
    """
    domain = Domain(boxlo=[0.0] * 3, boxhi=[BOX] * 3, boundary=("p", "p", "p"))
    atom = AtomSystem(max_atoms=2)
    atom.add_particles(
        x=np.array([[0.5 * BOX, 0.5 * BOX, 0.5 * BOX], [0.25 * BOX, 0.25 * BOX, 0.25 * BOX]]),
        radius=1e-4,
        density=DENSITY,
    )
    fix = FixDeformPressure(
        domain=domain,
        axes={d: {"style": "erate", "rate": -1.0e3} for d in "xyz"},
        nevery=1,
    )
    fix.setup(0)

    for _ in range(50):
        fix.end_of_step(atom, DT)

    prd = np.array([float(domain.prd[d]) for d in range(3)])
    lo = np.array([float(domain.boxlo[d]) for d in range(3)])
    pos = atom.x.to_numpy()[:2]

    assert prd[0] < BOX, "erate with a negative rate should shrink the box"
    # Fractional coordinates are what an affine remap preserves.
    np.testing.assert_allclose((pos[0] - lo) / prd, [0.5] * 3, atol=1e-12)
    np.testing.assert_allclose((pos[1] - lo) / prd, [0.25] * 3, atol=1e-12)
