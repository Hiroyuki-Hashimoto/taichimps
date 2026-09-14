"""
Tilted (restricted triclinic) periodic boxes, checked against the LAMMPS binary.

Domain used to be orthogonal-only, which meant shear could not be represented at
all. These tests cover the two places where a tilt actually changes the answer:
the minimum image convention, and the neighbor binning (taichimps has no ghost
atoms, so bins are laid out in lamda coordinates where wrapping a bin index
across a boundary is still a plain modulo).
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
from taichimps.computes.thermo import Computes
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
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


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def _config(seed: int, n_side: int = 3):
    """
    A jittered lattice of overlapping spheres, given in *lamda* coordinates.

    Positions are generated as fractions of the cell so the same configuration
    can be placed into an orthogonal or a tilted box and stay equivalent.
    """
    rng = np.random.default_rng(seed)
    lam = np.asarray(
        [
            [(a + 0.5) / n_side, (b + 0.5) / n_side, (c + 0.5) / n_side]
            for a in range(n_side)
            for b in range(n_side)
            for c in range(n_side)
        ],
        dtype=np.float64,
    )
    lam += rng.uniform(-0.05 / n_side, 0.05 / n_side, size=lam.shape)
    lam = np.mod(lam, 1.0)
    n = len(lam)
    radius = np.full(n, (BOX / n_side) * 0.52)
    density = np.full(n, DENSITY)
    v = rng.normal(scale=0.02, size=lam.shape)
    omega = rng.normal(scale=20.0, size=lam.shape)
    return lam, radius, density, v, omega


def _h(tilt) -> np.ndarray:
    """Shape matrix with the lattice vectors as columns."""
    xy, xz, yz = tilt
    return np.array([[BOX, xy, xz], [0.0, BOX, yz], [0.0, 0.0, BOX]])


def _cartesian(lam, tilt) -> np.ndarray:
    return lam @ _h(tilt).T


# Taichi specialises a kernel per field instance, so building a fresh
# Simulation for every case means recompiling everything each time -- minutes
# per case on a busy machine. One rig is built lazily and reset between cases
# instead; the tilt lives in a Domain field now, so even the box shape can be
# changed without touching the compiled code.
MAX_ATOMS = 64
_RIG: dict[str, object] = {}


def _rig():
    if not _RIG:
        domain = Domain(
            boxlo=[0.0, 0.0, 0.0],
            boxhi=[BOX, BOX, BOX],
            boundary=("p", "p", "p"),
        )
        atom = AtomSystem(max_atoms=MAX_ATOMS)
        neighbor = NeighborList(
            domain=domain, max_atoms=MAX_ATOMS, max_neighbors_per_atom=64, skin=SKIN
        )
        neighbor.check = False
        neighbor.every = 1
        history = ContactHistory(max_atoms=MAX_ATOMS, max_neighbors=64)
        pair = GranHookeHistory(
            domain=domain, kn=KN, gamman=GAMMAN, kt=KT, gammat=GAMMAT,
            xmu=XMU, dampflag=1,
        )
        sim = Simulation(
            domain=domain, atom=atom, neighbor=neighbor, history=history,
            pair=pair, dt=DT,
        )
        sim.add_fix(FixNVESphere(domain=domain))
        _RIG.update(domain=domain, atom=atom, neighbor=neighbor, history=history,
                    sim=sim)
    return _RIG


def _run_taichimps(lam, radius, density, v, omega, tilt, steps):
    rig = _rig()
    domain, atom, neighbor, history, sim = (
        rig["domain"], rig["atom"], rig["neighbor"], rig["history"], rig["sim"]
    )

    domain.set_box([0.0, 0.0, 0.0], [BOX, BOX, BOX], tilt=tilt)
    atom.nlocal = 0
    atom.add_particles(
        x=_cartesian(lam, tilt), radius=radius, density=density, v=v, omega=omega
    )
    history.reset()
    neighbor.has_built = False
    neighbor.last_build_step = -1
    neighbor.current_step = 0
    sim.timestep = 0

    n = atom.nlocal
    if steps == 0:
        sim.init_simulation()
    else:
        sim.run(steps)
    return sim, atom.f.to_numpy()[:n].copy(), atom.torque.to_numpy()[:n].copy()


def _assert_close(got, ref, label):
    scale = np.abs(ref).max()
    assert scale > 0.0, f"reference {label} is all zero; the case has no contacts"
    err = np.abs(got - ref).max() / scale
    assert err < 1e-9, (
        f"{label} differs from LAMMPS by {err:.3e} (relative to {scale:.3e})\n"
        f"taichimps[0:3]=\n{got[:3]}\nLAMMPS[0:3]=\n{ref[:3]}"
    )


@pytest.mark.skipif(lmp_executable() is None, reason="no LAMMPS executable available")
@pytest.mark.parametrize(
    "tilt",
    [
        (0.30 * BOX, 0.0, 0.0),
        (0.30 * BOX, -0.20 * BOX, 0.25 * BOX),
    ],
    ids=["xy_only", "xy_xz_yz"],
)
@pytest.mark.parametrize("steps", [0, 1, 25])
def test_tilted_box_forces_match_lammps(tmp_path, tilt, steps):
    """
    Forces and torques in a tilted cell, with contacts straddling the boundary.

    This is the end-to-end check on both the triclinic minimum image and the
    lamda-space binning: a wrong tilt correction on a boundary-crossing pair
    shows up immediately as a wrong force.
    """
    lam, radius, density, v, omega = _config(seed=17)
    x = _cartesian(lam, tilt)

    frames = run_lammps(
        tmp_path / f"lmp_{steps}",
        x=x,
        radius=radius,
        density=density,
        v=v,
        omega=omega,
        boxlo=(0.0, 0.0, 0.0),
        boxhi=(BOX, BOX, BOX),
        tilt=tilt,
        pair_style=f"gran/hooke/history {KN} {KT} {GAMMAN} {GAMMAT} {XMU} 1",
        steps=steps,
        dt=DT,
        skin=SKIN,
    )
    ref = frames[-1]

    _, f_got, tq_got = _run_taichimps(lam, radius, density, v, omega, tilt, steps)

    _assert_close(f_got, force_array(ref), "force")
    _assert_close(tq_got, torque_array(ref), "torque")


def test_shearing_the_cell_changes_the_stress():
    """
    A tilt must actually reach the physics, not be silently ignored.

    The same particles at the same fractional positions in a sheared cell are a
    genuinely different configuration, so the stress tensor has to differ -- and
    it must pick up off-diagonal (shear) components that the orthogonal cell
    does not have.
    """
    lam, radius, density, v, omega = _config(seed=23)
    computes = Computes()

    def stress(tilt):
        sim, _, _ = _run_taichimps(lam, radius, density, v, omega, tilt, 0)
        return computes.compute_pressure_tensor(sim.atom, sim.domain, kinetic=False)

    p_ortho = stress((0.0, 0.0, 0.0))
    p_shear = stress((0.25 * BOX, 0.0, 0.0))

    assert np.abs(p_ortho[:3]).max() > 0.0, "test packing produced no contacts"
    # xy is component 3 in the [xx, yy, zz, xy, xz, yz] ordering.
    assert abs(p_shear[3]) > 0.05 * np.abs(p_shear[:3]).max(), (
        "shearing the cell produced no xy shear stress, so the tilt is being ignored"
    )


def test_minimum_image_is_periodic_in_the_tilted_lattice():
    """
    Displacing a particle by any lattice vector must leave the forces unchanged.

    For a tilted cell the lattice vectors are b = (xy, yprd, 0) and
    c = (xz, yz, zprd), not the Cartesian axes, so this fails outright if the
    tilt corrections are missing from the minimum image convention.
    """
    tilt = (0.30 * BOX, -0.20 * BOX, 0.25 * BOX)
    lam, radius, density, v, omega = _config(seed=29)

    _, f_ref, tq_ref = _run_taichimps(lam, radius, density, v, omega, tilt, 0)

    # Shift every particle by a different lattice vector.
    shifted = lam.copy()
    shifted[0::3] += np.array([1.0, 0.0, 0.0])
    shifted[1::3] += np.array([0.0, -1.0, 0.0])
    shifted[2::3] += np.array([0.0, 0.0, 1.0])
    _, f_got, tq_got = _run_taichimps(shifted, radius, density, v, omega, tilt, 0)

    scale = np.abs(f_ref).max()
    assert scale > 0.0, "test packing produced no contacts"
    np.testing.assert_allclose(f_got, f_ref, atol=1e-9 * scale)
    np.testing.assert_allclose(tq_got, tq_ref, atol=1e-9 * np.abs(tq_ref).max())
