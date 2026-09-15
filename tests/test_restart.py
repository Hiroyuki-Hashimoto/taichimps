"""
LAMMPS-compatible restart files.

A restart must carry enough state that splitting a run in two reproduces the
uninterrupted run exactly: positions, velocities, angular velocities, the box,
and -- the part that is easy to lose -- the tangential contact history of every
touching pair.  Drop the history and the split run starts every contact from
zero shear, which for a frictional packing is a visible, growing divergence,
not a rounding difference.

The tests cross both ways: taichimps reads a restart written by the real LAMMPS
binary and continues from it, and LAMMPS reads one written by taichimps.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import taichi as ti
from lammps_harness import (
    DUMP_FIELDS,
    lmp_executable,
    parse_dump,
    run_lammps_script,
    write_data_file,
)

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.GRANULAR.granular import PairGranular
from taichimps.neighbor import NeighborList
from taichimps.restart import parse_neigh_history, read_restart, write_restart
from taichimps.simulation import Simulation

pytestmark = pytest.mark.skipif(
    lmp_executable() is None, reason="no LAMMPS binary to compare against"
)

# The material of the FCC triaxial case this project is validated against.
EMOD, CR, POIS, FRIC = 71.6e9, 0.95, 0.23, 0.3
RADIUS, DENSITY = 1.0e-3, 2500.0
DT = 1.0e-7
SKIN = 2.0e-4
HALF, TOTAL = 400, 800

PAIR_STYLE = "granular"
PAIR_COEFF = (
    f"* * hertz/material {EMOD:.17g} {CR:.17g} {POIS:.17g} "
    f"tangential mindlin NULL 1.0 {FRIC:.17g} damping coeff_restitution"
)


def _packing() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    A small overlapping FCC block with a spin, in a periodic box.

    The particles start compressed (lattice spacing below 2R) so that every
    particle has contacts from step zero, and spinning so that the tangential
    history grows rather than staying at machine zero.
    """
    n_cell = 3
    a = 2.0 * RADIUS * np.sqrt(2.0) * 0.985
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = [
        (np.array([ix, iy, iz]) + b) * a
        for ix in range(n_cell)
        for iy in range(n_cell)
        for iz in range(n_cell)
        for b in basis
    ]
    x = np.asarray(pts, dtype=np.float64)
    box = n_cell * a

    rng = np.random.default_rng(20260915)
    v = rng.normal(0.0, 0.02, size=x.shape)
    omega = rng.normal(0.0, 20.0, size=x.shape)
    return x, v, omega, box


def _lammps_preamble(box: float) -> str:
    return f"""units si
atom_style sphere
boundary p p p
newton off
dimension 3
comm_modify mode single vel yes
neighbor {SKIN:.17g} bin
neigh_modify delay 0 every 1 check no
"""


def _lammps_body(steps: int, dump_every: int, warmup: int = 0) -> str:
    """
    The run itself.

    `warmup` splits it into two `run` commands.  That is deliberate: LAMMPS
    re-runs Verlet::setup() at the start of every `run`, and setup evaluates the
    forces with `history_update = 0`, so `run 800` and `run 400` + `run 400` are
    *different trajectories*.  Only the latter is what a restart continues, so
    it is the reference a restart has to be measured against.
    """
    run_lines = f"run {warmup}\nrun {steps - warmup}\n" if warmup else f"run {steps}\n"
    return f"""pair_style {PAIR_STYLE}
pair_coeff {PAIR_COEFF}
timestep {DT:.17g}
fix 1 all nve/sphere
dump 1 all custom {dump_every} dump.out {' '.join(DUMP_FIELDS)}
dump_modify 1 sort id format float %.17e
thermo {steps if steps else 1}
""" + run_lines


def _write_inputs(workdir, x, v, omega, box):
    write_data_file(
        workdir / "data.in",
        x=x,
        radius=np.full(len(x), RADIUS),
        density=np.full(len(x), DENSITY),
        v=v,
        omega=omega,
        boxlo=(0.0, 0.0, 0.0),
        boxhi=(box, box, box),
    )


def _lammps_continuous(workdir, x, v, omega, box) -> dict[str, np.ndarray]:
    workdir.mkdir(parents=True, exist_ok=True)
    _write_inputs(workdir, x, v, omega, box)
    run_lammps_script(
        workdir,
        _lammps_preamble(box)
        + "read_data data.in\n"
        + _lammps_body(TOTAL, TOTAL, warmup=HALF),
        "in.cont",
    )
    return parse_dump(workdir / "dump.out")[-1]


def _lammps_first_half(workdir, x, v, omega, box) -> None:
    """Run the first half and leave `mid.restart` behind."""
    workdir.mkdir(parents=True, exist_ok=True)
    _write_inputs(workdir, x, v, omega, box)
    run_lammps_script(
        workdir,
        _lammps_preamble(box)
        + "read_data data.in\n"
        + _lammps_body(HALF, HALF)
        + "write_restart mid.restart\n",
        "in.first",
    )


def _lammps_second_half(workdir, box, restart_name: str) -> dict[str, np.ndarray]:
    """Resume from a restart file and run the second half."""
    run_lammps_script(
        workdir,
        _lammps_preamble(box)
        + f"read_restart {restart_name}\n"
        + "reset_timestep 0\n"
        + _lammps_body(TOTAL - HALF, TOTAL - HALF),
        "in.second",
    )
    return parse_dump(workdir / "dump.out")[-1]


def _assert_same_state(a, b, box, tol=1e-9, label=""):
    """
    Compare two end states field by field.

    Positions are compared as minimum-image separations and measured against
    the particle radius.  A coordinate is only meaningful modulo the box, and
    the two codes do not agree on which side of a periodic boundary an atom
    sitting exactly on it is reported: taichimps leaves it at boxhi where
    LAMMPS has already wrapped it to boxlo.  That is a whole box length of
    apparent error for a particle that has not moved at all.  Velocity and spin
    are measured against their own magnitude.
    """
    for field in ("x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz"):
        diff = np.abs(a[field] - b[field])
        if field in ("x", "y", "z"):
            diff = np.remainder(diff, box)
            diff = np.minimum(diff, box - diff)
            scale = RADIUS
        else:
            scale = np.abs(a[field]).max()
        err = diff.max() / max(scale, 1e-30)
        assert err < tol, f"{label}{field} differs by {err:.3e} (relative)"


def test_lammps_restart_reproduces_the_uninterrupted_run(tmp_path):
    """
    The reference: LAMMPS itself must be restart-exact for this case.

    If it is not, nothing downstream can be, so this pins the expectation the
    taichimps tests are held to.

    The comparison is against `run 400` + `run 400` rather than `run 800`,
    because a restart resumes the way a second `run` command does -- see
    _lammps_body.  What is left is only that the resumed run rebuilds its
    neighbor list from scratch, so contacts are summed in a different order and
    the last digits move.
    """
    x, v, omega, box = _packing()
    cont = _lammps_continuous(tmp_path / "cont", x, v, omega, box)
    split_dir = tmp_path / "split"
    _lammps_first_half(split_dir, x, v, omega, box)
    split = _lammps_second_half(split_dir, box, "mid.restart")
    _assert_same_state(cont, split, box, tol=1e-8,
                       label="LAMMPS split vs continuous: ")


# --------------------------------------------------------------------------
# taichimps side


def _make_sim(x, v, omega, box, n_max=None, history=None, radius=None,
              density=None, tag=None, boxlo=None, boxhi=None):
    n = len(x)
    n_max = n_max or n + 8
    boxlo = [0.0] * 3 if boxlo is None else list(boxlo)
    boxhi = [box] * 3 if boxhi is None else list(boxhi)
    domain = Domain(boxlo=boxlo, boxhi=boxhi, boundary=("p", "p", "p"))
    atom = AtomSystem(max_atoms=n_max)
    atom.add_particles(
        x=np.asarray(x, dtype=np.float64),
        radius=RADIUS if radius is None else radius,
        density=DENSITY if density is None else density,
        v=v,
        omega=omega,
        tag=tag,
    )
    neighbor = NeighborList(
        domain=domain, max_atoms=n_max, max_neighbors_per_atom=64, skin=SKIN
    )
    if history is None:
        history = ContactHistory(max_atoms=n_max, max_neighbors=64)
    pair = PairGranular(
        domain=domain,
        normal="hertz/material",
        normal_coeffs=[EMOD, CR, POIS],
        tangential="mindlin",
        tangential_coeffs=[None, 1.0, FRIC],
        damping="coeff_restitution",
    )
    sim = Simulation(
        domain=domain, atom=atom, neighbor=neighbor, history=history, pair=pair, dt=DT
    )
    sim.add_fix(FixNVESphere(domain=domain))
    return sim


def _state(sim) -> dict[str, np.ndarray]:
    n = sim.atom.nlocal
    order = np.argsort(sim.atom.tag.to_numpy()[:n])
    x = sim.atom.x.to_numpy()[:n][order]
    v = sim.atom.v.to_numpy()[:n][order]
    w = sim.atom.omega.to_numpy()[:n][order]
    return {
        "x": x[:, 0], "y": x[:, 1], "z": x[:, 2],
        "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2],
        "omegax": w[:, 0], "omegay": w[:, 1], "omegaz": w[:, 2],
    }


def _sim_from_restart(path):
    """Rebuild a Simulation from a restart file, history included."""
    data = read_restart(path)
    n_max = data.natoms + 8
    history = ContactHistory(max_atoms=n_max, max_neighbors=64)
    if data.fix_extra:
        decoded = [parse_neigh_history(rec) for rec in data.fix_extra]
        history.load_restart([t for t, _ in decoded], [s for _, s in decoded])
    sim = _make_sim(
        data.x, data.v, data.omega, box=None,
        n_max=n_max, history=history,
        radius=data.radius, density=data.density, tag=data.tag,
        boxlo=data.boxlo, boxhi=data.boxhi,
    )
    sim.timestep = data.timestep
    return sim, data


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def test_taichimps_restart_reproduces_its_own_uninterrupted_run(tmp_path):
    """Write a restart halfway through, resume from it, land in the same place."""
    x, v, omega, box = _packing()

    # Two run calls, for the reason given in _lammps_body: a restart resumes
    # the way a second `run` command does, not the way one long run does.
    cont = _make_sim(x, v, omega, box)
    cont.run(HALF)
    cont.run(TOTAL - HALF)

    first = _make_sim(x, v, omega, box)
    first.run(HALF)
    path = tmp_path / "mid.restart"
    write_restart(
        path, first.atom, first.domain, timestep=first.timestep, dt=DT,
        history=first.history, pair=first.pair_style,
    )

    second, _ = _sim_from_restart(path)
    second.run(TOTAL - HALF)

    _assert_same_state(_state(cont), _state(second), box, tol=1e-10,
                       label="taichimps split vs continuous: ")


def test_taichimps_continues_a_lammps_restart(tmp_path):
    """
    Read the real thing: LAMMPS runs the first half and writes the restart,
    taichimps reads it and runs the second half.

    This is what checks the binary reader against actual LAMMPS bytes, and it
    only passes if the contact history came across -- a frictional packing whose
    shear history is reset drifts well outside the tolerance in a few hundred
    steps.
    """
    x, v, omega, box = _packing()
    split_dir = tmp_path / "lmp"
    _lammps_first_half(split_dir, x, v, omega, box)
    reference = _lammps_continuous(tmp_path / "cont", x, v, omega, box)

    sim, data = _sim_from_restart(split_dir / "mid.restart")
    assert data.timestep == HALF
    assert data.pair_style == "granular"
    assert "NEIGH_HISTORY" in data.fix_peratom_styles
    npartner = np.array([parse_neigh_history(r)[0].size for r in data.fix_extra])
    assert npartner.sum() > 0, "the restart carried no contact history"

    sim.run(TOTAL - HALF)
    _assert_same_state(reference, _state(sim), box, tol=1e-6,
                       label="taichimps from a LAMMPS restart: ")


def test_lammps_continues_a_taichimps_restart(tmp_path):
    """The other direction: LAMMPS must accept what taichimps writes."""
    x, v, omega, box = _packing()
    reference = _lammps_continuous(tmp_path / "cont", x, v, omega, box)

    first = _make_sim(x, v, omega, box)
    first.run(HALF)
    workdir = tmp_path / "resume"
    workdir.mkdir(parents=True, exist_ok=True)
    write_restart(
        workdir / "mid.restart", first.atom, first.domain,
        timestep=first.timestep, dt=DT, history=first.history,
        pair=first.pair_style,
    )

    resumed = _lammps_second_half(workdir, box, "mid.restart")
    _assert_same_state(reference, resumed, box, tol=1e-6,
                       label="LAMMPS from a taichimps restart: ")


def test_restart_round_trip_is_bit_exact(tmp_path):
    """Everything taichimps writes, taichimps reads back unchanged."""
    x, v, omega, box = _packing()
    sim = _make_sim(x, v, omega, box)
    sim.run(HALF)

    path = tmp_path / "rt.restart"
    write_restart(path, sim.atom, sim.domain, timestep=sim.timestep, dt=DT,
                  history=sim.history, pair=sim.pair_style)
    data = read_restart(path)
    assert data.pair_style == "granular"
    model = data.pair_settings["models"][0]
    assert model["normal"] == ("hertz/material", [EMOD, CR, POIS])
    assert model["tangential"] == ("mindlin", [-1.0, 1.0, FRIC])
    assert model["damping"][0] == "coeff_restitution"

    n = sim.atom.nlocal
    assert data.timestep == sim.timestep
    assert data.dt == DT
    assert data.natoms == n
    np.testing.assert_array_equal(data.x, sim.atom.x.to_numpy()[:n])
    np.testing.assert_array_equal(data.v, sim.atom.v.to_numpy()[:n])
    np.testing.assert_array_equal(data.omega, sim.atom.omega.to_numpy()[:n])
    np.testing.assert_array_equal(data.radius, sim.atom.radius.to_numpy()[:n])
    np.testing.assert_array_equal(data.rmass, sim.atom.rmass.to_numpy()[:n])
    np.testing.assert_array_equal(data.tag, sim.atom.tag.to_numpy()[:n])

    # Every contact must appear under both partners, the second copy negated,
    # which is how LAMMPS stores it -- see restart._both_sided_history.
    partner = sim.history.partner.to_numpy()[:n]
    shear = sim.history.shear.to_numpy()[:n]
    tag_index = {int(t): i for i, t in enumerate(sim.atom.tag.to_numpy()[:n])}
    stored = []
    for i in range(n):
        tags, vals = parse_neigh_history(data.fix_extra[i])
        stored.append({int(t): vals[k] for k, t in enumerate(tags)})

    total = 0
    for i in range(n):
        for k in np.nonzero(partner[i] >= 0)[0]:
            ptag = int(partner[i][k])
            j = tag_index[ptag]
            itag = int(sim.atom.tag.to_numpy()[i])
            np.testing.assert_array_equal(stored[i][ptag], shear[i][k])
            np.testing.assert_array_equal(stored[j][itag], -shear[i][k])
            total += 1
    assert total > 0, "the run produced no contacts to store"
    assert sum(len(d) for d in stored) == 2 * total


# --------------------------------------------------------------------------
# the input-script commands


_RUNNER = (
    "import sys; from taichimps.input import parse_and_run; parse_and_run(sys.argv[1])"
)


def _run_script(path: Path) -> None:
    """
    Execute a taichimps input script in its own process.

    The parser calls ti.init(), which tears down every Taichi field that already
    exists, so it cannot share a process with the tests above.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(path)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{path.name} failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
        )


def test_the_restart_commands_work_from_an_input_script(tmp_path):
    """
    `write_restart` / `read_restart` as script commands, end to end.

    The resumed script names no pair style and no timestep: both have to come
    back out of the restart file, the way `read_restart` works in LAMMPS.
    """
    x, v, omega, box = _packing()
    workdir = tmp_path / "script"
    workdir.mkdir(parents=True, exist_ok=True)
    _write_inputs(workdir, x, v, omega, box)

    common = f"""units si
atom_style sphere
boundary p p p
newton off
dimension 3
neighbor {SKIN:.17g} bin
"""
    pair = f"""pair_style {PAIR_STYLE}
pair_coeff {PAIR_COEFF}
timestep {DT:.17g}
"""
    (workdir / "in.first").write_text(
        common + "read_data data.in\n" + pair + f"""fix 1 all nve/sphere
run {HALF}
write_restart mid.restart
run {TOTAL - HALF}
write_restart cont.restart
"""
    )
    (workdir / "in.second").write_text(
        common + "read_restart mid.restart\n" + f"""fix 1 all nve/sphere
run {TOTAL - HALF}
write_restart split.restart
"""
    )
    _run_script(workdir / "in.first")
    _run_script(workdir / "in.second")

    mid = read_restart(workdir / "mid.restart")
    assert mid.timestep == HALF
    assert mid.dt == DT
    assert mid.pair_style == "granular"

    cont = read_restart(workdir / "cont.restart")
    split = read_restart(workdir / "split.restart")
    assert split.timestep == cont.timestep == TOTAL

    order_c = np.argsort(cont.tag)
    order_s = np.argsort(split.tag)
    for name in ("x", "v", "omega"):
        a = getattr(cont, name)[order_c]
        b = getattr(split, name)[order_s]
        if name == "x":
            diff = np.remainder(np.abs(a - b), box)
            diff = np.minimum(diff, box - diff)
            err = diff.max() / RADIUS
        else:
            err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
        assert err < 1e-10, f"script restart: {name} differs by {err:.3e}"
