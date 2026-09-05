from pathlib import Path

from taichimps.data_reader import read_data


def test_read_data_lj():
    p = Path("/home/kmakoto/Desktop/lammps_work/ppp_x0.058611_y0.058611_z0.058611_e1.4.lj")
    if not p.exists():
        return
    data = read_data(p)
    assert data.natoms == 30000
    assert len(data.x) == 30000
    assert data.boxlo == [0.0, 0.0, 0.0]
    assert abs(data.boxhi[0] - 0.058611) < 1e-6
    assert data.radius[0] > 0.0
    assert data.density[0] == 2700.0
