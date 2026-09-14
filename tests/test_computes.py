"""
Named computes, checked against the LAMMPS binary where possible.

The parser used to ignore `compute` entirely and fill in a handful of hard-coded
names (c_1, c_3, c_5, c_6) with meanings taken from one particular reference
script. A script that numbered its computes differently read zeros, `c_1[1]`
never evaluated as an expression at all (it raised NameError inside the
evaluator and the raw string came back), and the `NULL` that drops the kinetic
term had no effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent))

from lammps_harness import lmp_executable, run_lammps

from taichimps.input import LAMMPSInputParser

BOX = 0.02
DENSITY = 2650.0
KN, KT = 1.0e6, 2.0e5
GAMMAN, GAMMAT = 500.0, 250.0
XMU = 0.5
DT = 1.0e-6
SKIN = 0.0005
N_SIDE = 3


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def _config(seed: int = 3):
    rng = np.random.default_rng(seed)
    spacing = BOX / N_SIDE
    x = np.asarray(
        [
            [(a + 0.5) * spacing, (b + 0.5) * spacing, (c + 0.5) * spacing]
            for a in range(N_SIDE)
            for b in range(N_SIDE)
            for c in range(N_SIDE)
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


def _write_data(path: Path, cfg) -> None:
    from lammps_harness import write_data_file

    x, radius, density, v, omega = cfg
    write_data_file(
        path, x, radius, density, v, omega, (0.0, 0.0, 0.0), (BOX, BOX, BOX)
    )


SCRIPT = f"""
units           si
boundary        p p p
atom_style      sphere
dimension       3
newton          off

read_data       data.in

pair_style      gran/hooke/history {KN} {KT} {GAMMAN} {GAMMAT} {XMU} 1

neighbor        {SKIN} bin
neigh_modify    delay 0 every 1 check no
timestep        {DT}

fix             2 all nve/sphere

compute         1 all pressure NULL pair
compute         2 all contact/atom
compute         3 all reduce ave c_2
compute         5 all ke
compute         6 all erotate/sphere
variable        pvol atom "4.0*PI*radius*radius*radius/3.0"
compute         4 all reduce sum v_pvol

variable        pxx equal 'c_1[1]'
variable        pyy equal 'c_1[2]'
variable        pzz equal 'c_1[3]'
variable        cn  equal c_3
variable        solid equal c_4

run             5
"""


def _run_parser(tmp_path, cfg) -> LAMMPSInputParser:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_data(tmp_path / "data.in", cfg)
    script = tmp_path / "in.computes"
    script.write_text(SCRIPT)
    parser = LAMMPSInputParser(script)
    parser.execute()
    parser.update_dynamic_variables()
    return parser


def test_compute_vector_reference_resolves(tmp_path):
    """
    `c_1[1]` must come back as a number.

    Before computes were parsed this raised NameError inside the expression
    evaluator, which swallowed it and returned the literal string 'c_1[1]'.
    """
    cfg = _config()
    parser = _run_parser(tmp_path, cfg)

    for name in ("pxx", "pyy", "pzz"):
        value = parser.evaluate_variable(name)
        assert isinstance(value, float)
    # A contacting pack has a non-zero pressure on every axis.
    assert abs(parser.evaluate_variable("pxx")) > 0.0


def test_reduce_over_an_atom_variable(tmp_path):
    """`compute reduce sum v_pvol` over an atom-style variable."""
    cfg = _config()
    _x, radius, _density, _v, _omega = cfg
    parser = _run_parser(tmp_path, cfg)

    expected = float(np.sum(4.0 / 3.0 * np.pi * radius**3))
    np.testing.assert_allclose(parser.evaluate_variable("solid"), expected, rtol=1e-12)


def test_unknown_compute_style_is_rejected(tmp_path):
    """An unimplemented style must fail loudly rather than silently read zero."""
    _write_data(tmp_path / "data.in", _config())
    script = tmp_path / "in.bad"
    script.write_text(
        "units si\nboundary p p p\natom_style sphere\nread_data data.in\n"
        "compute 9 all rdf 100\n"
    )
    parser = LAMMPSInputParser(script)
    with pytest.raises(ValueError, match="rdf"):
        parser.execute()


@pytest.mark.skipif(lmp_executable() is None, reason="no LAMMPS executable available")
def test_computes_match_lammps(tmp_path, capsys):
    """
    Compare pressure, coordination number and the energies against LAMMPS.

    `compute pressure NULL pair` in particular has to drop the kinetic term;
    reading it with the kinetic term included was what the parser used to do.
    """
    cfg = _config()
    x, radius, density, v, omega = cfg
    steps = 5

    run_lammps(
        tmp_path / "lmp",
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
        pre_thermo=(
            "compute cn all contact/atom\n"
            "compute cnave all reduce ave c_cn\n"
            "compute kt all ke\n"
            "compute kr all erotate/sphere"
        ),
        thermo_extra="c_cnave c_kt c_kr",
    )
    log = (tmp_path / "lmp" / "log.lammps").read_text().splitlines()
    header = next(i for i, line in enumerate(log) if line.split()[:1] == ["Step"])
    last = log[header + steps + 1].split()
    # step lx ly lz c_vir[1..3] c_cnave c_kt c_kr
    p_ref = np.array([float(t) for t in last[4:7]])
    cn_ref, ke_ref, erot_ref = (float(t) for t in last[7:10])

    parser = _run_parser(tmp_path / "tm", cfg)
    p_got = np.array(
        [parser.evaluate_variable(n) for n in ("pxx", "pyy", "pzz")]
    )

    # This test is about the compute plumbing, not force parity -- that is
    # covered at 1e-9 in test_lammps_reference.py, which feeds both engines the
    # same arrays directly. Here taichimps reads its state back from a data
    # file, so the two trajectories differ at round-off and a stiff contact
    # system amplifies that over five steps (forces agree to ~4e-10, the
    # velocity-squared energies to ~5e-9).
    rtol = 1e-7

    assert np.abs(p_ref).max() > 0.0, "the reference pack produced no contacts"
    np.testing.assert_allclose(p_got, p_ref, rtol=rtol)
    # Coordination number is an integer count per particle, so this one is exact.
    np.testing.assert_allclose(parser.evaluate_variable("cn"), cn_ref, rtol=1e-12)

    ctx = parser.compute_context()
    np.testing.assert_allclose(parser.computes["5"].scalar(ctx), ke_ref, rtol=rtol)
    np.testing.assert_allclose(parser.computes["6"].scalar(ctx), erot_ref, rtol=rtol)
