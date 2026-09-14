"""
fix wall/gran, checked against the LAMMPS binary.

The wall used to be a viscous tangential dashpot with a Coulomb cap and no
shear history at all, i.e. the history-free `hooke` style, so a specimen resting
against walls had no frictional memory at the boundary.  These tests pin both
the history-free and the history variants against LAMMPS, with particles that
are both sliding and spinning against the wall.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent))

from lammps_harness import force_array, lmp_executable, run_lammps, torque_array

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.EXTRA_FIX.wall_gran import FixWallGran
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation

pytestmark = pytest.mark.skipif(
    lmp_executable() is None, reason="no LAMMPS executable available"
)

BOX = 0.02
DENSITY = 2650.0
RADIUS = 0.002
KN, KT = 1.0e6, 2.0e5
GAMMAN, GAMMAT = 500.0, 250.0
XMU = 0.5
# Well inside the Hookean contact time pi*sqrt(m/kn) ~ 30 us, so the particles
# stay on the wall for the whole 20-step case instead of rebounding off it.
DT = 2.0e-7
WALL_Z = 0.0


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def _config(seed: int = 5):
    """
    Particles pressed into the z = 0 wall, sliding and spinning.

    They sit on a regular 4x3 grid so that no two of them ever touch, in the
    periodic x/y directions either: the test is about the wall, and a stray
    pair contact would mix pair forces into the comparison.
    """
    rng = np.random.default_rng(seed)
    nx, ny = 4, 3
    xs = np.array(
        [[(a + 0.5) * BOX / nx, (b + 0.5) * BOX / ny] for a in range(nx) for b in range(ny)]
    )
    n = len(xs)
    # Overlap the wall by 2-20% of the radius.
    z = RADIUS * rng.uniform(0.80, 0.98, size=(n, 1))
    x = np.hstack([xs, z])

    spacing = min(BOX / nx, BOX / ny)
    assert spacing > 2.0 * RADIUS + 0.0005, "grid is too tight, particles would touch"

    v = rng.normal(scale=0.05, size=(n, 3))
    v[:, 2] = -np.abs(v[:, 2])  # pressing into the wall
    omega = rng.normal(scale=30.0, size=(n, 3))
    return x, np.full(n, RADIUS), np.full(n, DENSITY), v, omega


def _run_taichimps(cfg, steps, history):
    x, radius, density, v, omega = cfg
    domain = Domain(
        boxlo=[0.0, 0.0, 0.0], boxhi=[BOX, BOX, BOX], boundary=("p", "p", "f")
    )
    n = len(x)
    atom = AtomSystem(max_atoms=n)
    atom.add_particles(x=x, radius=radius, density=density, v=v, omega=omega)
    neighbor = NeighborList(domain=domain, max_atoms=n, max_neighbors_per_atom=32,
                            skin=0.0005)
    neighbor.check = False
    neighbor.every = 1
    sim = Simulation(domain=domain, atom=atom, neighbor=neighbor, pair=None, dt=DT)
    sim.add_fix(FixNVESphere(domain=domain))
    sim.add_fix(
        FixWallGran(
            domain=domain,
            wall_axis=2,
            wall_side=-1,
            wall_coord=WALL_Z,
            kn=KN,
            gamman=GAMMAN,
            kt=KT,
            gammat=GAMMAT,
            xmu=XMU,
            history=history,
            max_atoms=n,
        )
    )
    if steps == 0:
        sim.init_simulation()
    else:
        sim.run(steps)
    return atom.f.to_numpy()[:n].copy(), atom.torque.to_numpy()[:n].copy()


def _run_lammps(tmp_path, cfg, steps, style):
    x, radius, density, v, omega = cfg
    # A pair style is mandatory, so use one that cannot act: the particles are
    # far enough apart that no pair is ever in contact, leaving only the wall.
    frames = run_lammps(
        tmp_path,
        x=x,
        radius=radius,
        density=density,
        v=v,
        omega=omega,
        boxlo=(0.0, 0.0, 0.0),
        boxhi=(BOX, BOX, BOX),
        pair_style=f"gran/hooke {KN} {KT} {GAMMAN} {GAMMAT} {XMU} 1",
        steps=steps,
        dt=DT,
        skin=0.0005,
        boundary="p p f",
        extra_fixes=(
            f"fix w all wall/gran {style} {KN} {KT} {GAMMAN} {GAMMAT} {XMU} 1 "
            f"zplane {WALL_Z} NULL"
        ),
    )
    return frames[-1]


def _assert_close(got, ref, label):
    scale = np.abs(ref).max()
    assert scale > 0.0, f"reference {label} is all zero; no wall contact in this case"
    err = np.abs(got - ref).max() / scale
    assert err < 1e-9, (
        f"{label} differs from LAMMPS by {err:.3e} (relative to {scale:.3e})\n"
        f"taichimps[0:3]=\n{got[:3]}\nLAMMPS[0:3]=\n{ref[:3]}"
    )


@pytest.mark.parametrize("steps", [0, 1, 20])
def test_wall_hooke_history_matches_lammps(tmp_path, steps):
    """The history variant, where the tangential force is dominated by the shear."""
    cfg = _config()
    ref = _run_lammps(tmp_path / f"lmp_h_{steps}", cfg, steps, "hooke/history")
    f_got, tq_got = _run_taichimps(cfg, steps, history=True)
    _assert_close(f_got, force_array(ref), "force")
    _assert_close(tq_got, torque_array(ref), "torque")


def test_wall_hooke_nohistory_matches_lammps(tmp_path):
    """The history-free variant, which is what this fix used to implement."""
    cfg = _config(seed=9)
    ref = _run_lammps(tmp_path / "lmp_nh", cfg, 20, "hooke")
    f_got, tq_got = _run_taichimps(cfg, 20, history=False)
    _assert_close(f_got, force_array(ref), "force")
    _assert_close(tq_got, torque_array(ref), "torque")
