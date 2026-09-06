import numpy as np
import pytest
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_TAICHI import FixProbe, FixWave, WinSponge


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def test_fix_wave_force():
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0], boundary=["p", "p", "p"])
    atom = AtomSystem(max_atoms=10, float_type=ti.f64)
    atom.add_particles(
        x=[[5.0, 5.0, 1.0]],
        v=[[0.0, 0.0, 0.0]],
        omega=[[0.0, 0.0, 0.0]],
        radius=[0.5],
        density=1000.0,
        atom_type=1,
    )

    wave = FixWave(
        domain=domain,
        axis="z",
        waveform="pulse",
        f0=50.0,
        amplitude=100.0,
        mode="force",
        float_type=ti.f64,
    )

    # For pulse waveform with f0=50, period is 1/50 = 0.02s
    # Peak is at t = 0.5 * period = 0.01s (sin(pi * 50 * 0.01) = sin(pi/2) = 1.0)
    dt = 0.01
    wave.post_force(atom, dt)

    f_np = atom.f.to_numpy()
    # Pulse amplitude at peak = 100.0
    assert np.isclose(f_np[0, 2], 100.0, atol=1e-3)
    assert np.isclose(f_np[0, 0], 0.0)


def test_win_sponge():
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0], boundary=["f", "f", "f"])
    atom = AtomSystem(max_atoms=10, float_type=ti.f64)
    # Particle near boundary x=0.5 (within sponge thickness 2.0)
    atom.add_particles(
        x=[[0.5, 5.0, 5.0]],
        v=[[10.0, 0.0, 0.0]],
        omega=[[0.0, 0.0, 0.0]],
        radius=[0.5],
        density=1000.0,
        atom_type=1,
    )

    sponge = WinSponge(
        domain=domain,
        thickness=2.0,
        eta_max=100.0,
        power=2.0,
        boundaries=["xlo"],
        float_type=ti.f64,
    )

    # Initial force is 0
    atom.f[0] = ti.Vector([0.0, 0.0, 0.0])
    sponge.post_force(atom, 0.001)

    f_np = atom.f.to_numpy()
    # Damping force should oppose velocity in x
    assert f_np[0, 0] < 0.0


def test_fix_probe():
    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[10.0, 10.0, 10.0], boundary=["p", "p", "p"])
    atom = AtomSystem(max_atoms=10, float_type=ti.f64)
    atom.add_particles(
        x=[[5.0, 5.0, 5.0]],
        v=[[1.5, -2.0, 3.5]],
        omega=[[0.0, 0.0, 0.0]],
        radius=[0.5],
        density=1000.0,
        atom_type=1,
    )

    probe = FixProbe(
        domain=domain,
        points=[[5.0, 5.0, 5.0]],
        radius=1.0,
        nevery=1,
        float_type=ti.f64,
    )

    probe.end_of_step(atom, 0.001)
    data = probe.to_numpy()

    assert data.shape[0] == 1  # 1 step
    assert data.shape[1] == 1  # 1 probe
    assert np.isclose(data[0, 0, 0], 1.5)  # vx
    assert np.isclose(data[0, 0, 1], -2.0)  # vy
    assert np.isclose(data[0, 0, 2], 3.5)  # vz
