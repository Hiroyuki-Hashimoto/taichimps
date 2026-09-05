from pathlib import Path

import taichi as ti

from taichimps.input import LAMMPSInputParser


def test_isotropic_run():
    script_path = Path("/home/kmakoto/Desktop/lammps_work/in.isotropic_test")
    if not script_path.exists():
        return
    ti.init(arch=ti.cpu)
    parser = LAMMPSInputParser(script_path)
    parser.execute()

    assert parser.simulation is not None
    assert parser.simulation.timestep == 200
    assert parser.atom is not None
    assert parser.atom.nlocal == 30000

    out_file = script_path.parent / "mean_stress_test.txt"
    assert out_file.exists()
    with open(out_file, "r") as f:
        lines = f.readlines()
        assert len(lines) >= 3
