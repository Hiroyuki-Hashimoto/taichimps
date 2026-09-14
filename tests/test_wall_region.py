import numpy as np
import pytest
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.wall_gran_region import FixWallGranRegion
from taichimps.input import LAMMPSInputParser


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    ti.init(arch=ti.cpu)


def test_fix_wall_gran_region_cylinder_in():
    # Cylinder along Z-axis at (c1=0, c2=0), radius R=1.0, side="in"
    domain = Domain(boxlo=[-2.0, -2.0, -2.0], boxhi=[2.0, 2.0, 2.0])
    atom = AtomSystem(max_atoms=10)

    # Particle 0: inside cylinder near wall at x=0.95, y=0.0, z=0.0, radius=0.1
    # Distance to axis r_perp = 0.95. Wall is at R=1.0.
    # Overlap delta = r_i - (R - r_perp) = 0.1 - (1.0 - 0.95) = 0.05 > 0
    # Normal points inward: (-1, 0, 0)
    # Expected elastic force: kn * delta * (-1, 0, 0)
    kn = 1e4
    gamman = 10.0
    fix = FixWallGranRegion(
        domain=domain,
        axis=2,  # z-axis
        c1=0.0,
        c2=0.0,
        radius=1.0,
        side="in",
        fstyle="hooke",
        kn=kn,
        gamman=gamman,
        kt=5000.0,
        gammat=5.0,
        xmu=0.5,
        dampflag=0,
    )

    x_np = np.zeros((10, 3), dtype=np.float64)
    v_np = np.zeros((10, 3), dtype=np.float64)
    f_np = np.zeros((10, 3), dtype=np.float64)
    omega_np = np.zeros((10, 3), dtype=np.float64)
    torque_np = np.zeros((10, 3), dtype=np.float64)
    radius_np = np.full(10, 0.1, dtype=np.float64)
    rmass_np = np.full(10, 1.0, dtype=np.float64)

    x_np[0] = [0.95, 0.0, 0.0]
    # Moving towards wall (+x) -> normal velocity is vn = v . (-1, 0, 0) = -1.0
    v_np[0] = [1.0, 0.0, 0.0]

    atom.nlocal = 1
    atom.x.from_numpy(x_np)
    atom.v.from_numpy(v_np)
    atom.f.from_numpy(f_np)
    atom.omega.from_numpy(omega_np)
    atom.torque.from_numpy(torque_np)
    atom.radius.from_numpy(radius_np)
    atom.rmass.from_numpy(rmass_np)

    dt = 1e-4
    fix.post_force(atom, dt)

    f_res = atom.f.to_numpy()[0]
    # Elastic force: kn * delta = 1e4 * 0.05 = 500.0 along (-1, 0, 0) -> fx_elastic = -500.0
    # Damping: vn = vr . n = 1.0 * (-1) = -1.0
    # fn = fn_elastic - gamman * vn = 500.0 - 10.0 * (-1.0) = 510.0
    # Force on particle = fn * n = 510.0 * (-1, 0, 0) = (-510, 0, 0)
    assert np.isclose(f_res[0], -510.0, rtol=1e-4)
    assert np.isclose(f_res[1], 0.0, atol=1e-5)
    assert np.isclose(f_res[2], 0.0, atol=1e-5)


def test_fix_wall_gran_region_cylinder_out():
    # Cylinder along Z-axis, radius R=1.0, side="out" (particles outside)
    domain = Domain(boxlo=[-2.0, -2.0, -2.0], boxhi=[2.0, 2.0, 2.0])
    atom = AtomSystem(max_atoms=10)

    # Particle at x=1.05, y=0, z=0, radius=0.1
    # r_perp = 1.05. Overlap delta = r_i - (r_perp - R) = 0.1 - (1.05 - 1.0) = 0.05 > 0
    # Normal points outward: (+1, 0, 0)
    kn = 1e4
    fix = FixWallGranRegion(
        domain=domain,
        axis="z",
        c1=0.0,
        c2=0.0,
        radius=1.0,
        side="out",
        fstyle="hooke",
        kn=kn,
        gamman=0.0,
    )

    x_np = np.zeros((10, 3), dtype=np.float64)
    x_np[0] = [1.05, 0.0, 0.0]
    atom.nlocal = 1
    atom.x.from_numpy(x_np)
    atom.v.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.f.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.omega.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.torque.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.radius.from_numpy(np.full(10, 0.1, dtype=np.float64))
    atom.rmass.from_numpy(np.full(10, 1.0, dtype=np.float64))

    fix.post_force(atom, 1e-4)
    f_res = atom.f.to_numpy()[0]
    # Elastic force: kn * delta = 1e4 * 0.05 = 500.0 along (+1, 0, 0)
    assert np.isclose(f_res[0], 500.0, rtol=1e-4)


def test_fix_wall_gran_region_friction_and_torque():
    # Cylinder along Z-axis, particle with tangential velocity and rotation
    domain = Domain(boxlo=[-2.0, -2.0, -2.0], boxhi=[2.0, 2.0, 2.0])
    atom = AtomSystem(max_atoms=10)

    # Particle at x=0.95, y=0.0, z=0.0, radius=0.1
    # Normal is (-1, 0, 0)
    # Velocity in y direction: v_y = 1.0
    # Tangential damping opposing v_y: ft in -y direction
    # Torque = -r * n x ft
    kn = 1e4
    gammat = 20.0
    xmu = 0.5
    fix = FixWallGranRegion(
        domain=domain,
        axis=2,
        c1=0.0,
        c2=0.0,
        radius=1.0,
        side="in",
        fstyle="hooke",
        kn=kn,
        gamman=0.0,
        kt=0.0,
        gammat=gammat,
        xmu=xmu,
        # In LAMMPS dampflag = 0 zeroes the tangential damping coefficient
        # outright; it does not mean "skip the meff factor", which is what this
        # test previously assumed. rmass is 1.0 here, so with dampflag = 1 the
        # meff scaling is invisible and the expected values below still hold.
        dampflag=1,
    )

    x_np = np.zeros((10, 3), dtype=np.float64)
    v_np = np.zeros((10, 3), dtype=np.float64)
    radius_np = np.full(10, 0.1, dtype=np.float64)
    rmass_np = np.full(10, 1.0, dtype=np.float64)

    x_np[0] = [0.95, 0.0, 0.0]
    v_np[0] = [0.0, 1.0, 0.0]  # tangential velocity

    atom.nlocal = 1
    atom.x.from_numpy(x_np)
    atom.v.from_numpy(v_np)
    atom.f.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.omega.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.torque.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.radius.from_numpy(radius_np)
    atom.rmass.from_numpy(rmass_np)

    fix.post_force(atom, 1e-4)

    f_res = atom.f.to_numpy()[0]
    torque_res = atom.torque.to_numpy()[0]

    # Normal force: fn = kn * delta = 10000 * 0.05 = 500.0, n = (-1, 0, 0) -> fx = -500.0
    assert np.isclose(f_res[0], -500.0, rtol=1e-4)

    # Tangential: vr = (0, 1, 0), vrt = (0, 1, 0)
    # ft_damp = gammat * vrt = 20 * 1.0 = 20.0
    # Coulomb limit: xmu * fn = 0.5 * 500 = 250.0 > 20.0, so ft_vec = (0, -20.0, 0)
    assert np.isclose(f_res[1], -20.0, rtol=1e-4)

    # Torque:
    # Contact offset from particle center: r_c = -r_i * n = -0.1 * (-1, 0, 0) = (0.1, 0, 0)
    # Torque = r_c x ft = (0.1, 0, 0) x (0, -20, 0) = (0, 0, -2.0)
    assert np.isclose(torque_res[2], -2.0, rtol=1e-4)


def test_fix_wall_gran_region_hertz():
    # Hertz model test
    domain = Domain(boxlo=[-2.0, -2.0, -2.0], boxhi=[2.0, 2.0, 2.0])
    atom = AtomSystem(max_atoms=10)

    # Particle at x=0.95, y=0.0, z=0.0, radius=0.1
    # delta = 0.05
    # polyhertz = sqrt(R_eff * delta) = sqrt(0.1 * 0.05) = sqrt(0.005)
    kn = 1e4
    fix = FixWallGranRegion(
        domain=domain,
        axis="z",
        c1=0.0,
        c2=0.0,
        radius=1.0,
        side="in",
        fstyle="hertz",
        kn=kn,
        gamman=0.0,
    )

    x_np = np.zeros((10, 3), dtype=np.float64)
    x_np[0] = [0.95, 0.0, 0.0]
    atom.nlocal = 1
    atom.x.from_numpy(x_np)
    atom.v.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.f.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.omega.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.torque.from_numpy(np.zeros((10, 3), dtype=np.float64))
    atom.radius.from_numpy(np.full(10, 0.1, dtype=np.float64))
    atom.rmass.from_numpy(np.full(10, 1.0, dtype=np.float64))

    fix.post_force(atom, 1e-4)
    f_res = atom.f.to_numpy()[0]
    expected_fn = kn * 0.05 * np.sqrt(0.1 * 0.05)
    assert np.isclose(f_res[0], -expected_fn, rtol=1e-4)


def test_input_parser_wall_gran_region(tmp_path):
    # Test LAMMPS script with region cylinder and fix wall/gran/region
    script_file = tmp_path / "in.cyl_region"
    script = """
    units lj
    atom_style sphere
    dimension 3
    boundary f f f

    region reg_cyl cylinder z 0.0 0.0 1.0 -2.0 2.0 side in
    create_box 1 reg_cyl

    fix 1 all wall/gran/region hooke 10000.0 NULL 10.0 NULL 0.5 0 region reg_cyl
    """
    script_file.write_text(script)

    parser = LAMMPSInputParser(script_file)
    parser.execute()

    assert "1" in parser.fixes
    fix = parser.fixes["1"]
    assert isinstance(fix, FixWallGranRegion)
    assert fix.axis == 2
    assert fix.radius == 1.0
    assert fix.side == -1
    assert fix.kn == 10000.0
    assert fix.gamman == 10.0
    assert fix.xmu == 0.5


def test_input_parser_wall_gran_cylinder_direct(tmp_path):
    # Test LAMMPS script with fix wall/gran ... cylinder <axis> <c1> <c2> <radius>
    script_file = tmp_path / "in.cyl_direct"
    script = """
    units lj
    atom_style sphere
    dimension 3
    boundary f f f

    region box block -2 2 -2 2 -2 2
    create_box 1 box

    fix 2 all wall/gran hooke 20000.0 NULL 5.0 NULL 0.3 0 cylinder y 0.5 -0.5 1.5
    """
    script_file.write_text(script)

    parser = LAMMPSInputParser(script_file)
    parser.execute()

    assert "2" in parser.fixes
    fix = parser.fixes["2"]
    assert isinstance(fix, FixWallGranRegion)
    assert fix.axis == 1  # y-axis
    assert fix.c1 == 0.5
    assert fix.c2 == -0.5
    assert fix.radius == 1.5
    assert fix.kn == 20000.0
    assert fix.gamman == 5.0
    assert fix.xmu == 0.3
