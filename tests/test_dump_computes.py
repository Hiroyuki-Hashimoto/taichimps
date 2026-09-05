"""
Tests for Dump and Computes (Thermo).
"""

import os
import tempfile

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.computes import Computes
from taichimps.domain import Domain
from taichimps.dump import DumpWriter


def test_computes_thermo():
    ti.init(arch=ti.cpu)
    atom = AtomSystem(max_atoms=10)
    computes = Computes()

    atom.add_particles(
        tag=[1, 2],
        atom_type=[1, 1],
        x=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        v=[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        omega=[[0.0, 0.0, 10.0], [5.0, 0.0, 0.0]],
        radius=[0.1, 0.1],
        density=[1000.0, 1000.0],
    )

    # Particle 1: m1 = 4/3 * pi * 0.001 * 1000 = 4.18879 kg
    # ke_t1 = 0.5 * m1 * 4 = 2 * m1
    # Particle 2: m2 = m1
    # ke_t2 = 0.5 * m2 * 16 = 8 * m2
    # total ke_t = 10 * m1
    m1 = 4.0 / 3.0 * np.pi * (0.1**3) * 1000.0
    expected_ke_trans = 10.0 * m1
    computed_ke_trans = computes.ke_trans(atom)
    np.testing.assert_allclose(computed_ke_trans, expected_ke_trans, rtol=1e-4)

    # ke_rot = 0.5 * (0.4 * m * r^2) * omega^2
    # I = 0.4 * m1 * 0.01 = 0.004 * m1
    # p1: 0.5 * I * 100 = 50 * I
    # p2: 0.5 * I * 25 = 12.5 * I
    # total = 62.5 * I
    expected_ke_rot = 62.5 * (0.4 * m1 * 0.01)
    computed_ke_rot = computes.ke_rot(atom)
    np.testing.assert_allclose(computed_ke_rot, expected_ke_rot, rtol=1e-4)


def test_dump_writer():
    ti.init(arch=ti.cpu)
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[1.0, 1.0, 1.0])
    atom = AtomSystem(max_atoms=10)
    atom.add_particles(
        tag=[1],
        atom_type=[1],
        x=[[0.5, 0.5, 0.5]],
        v=[[1.0, 0.0, 0.0]],
        radius=[0.05],
        density=[1000.0],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_file = os.path.join(tmpdir, "dump.atom")
        writer = DumpWriter(dump_file)
        writer.write_timestep(timestep=0, atom=atom, domain=domain)

        with open(dump_file, "r") as f:
            content = f.read()

        assert "ITEM: TIMESTEP" in content
        assert "ITEM: NUMBER OF ATOMS" in content
        assert "1" in content
        assert "ITEM: BOX BOUNDS" in content
        assert "ITEM: ATOMS id type x y z vx vy vz omegax omegay omegaz radius" in content
