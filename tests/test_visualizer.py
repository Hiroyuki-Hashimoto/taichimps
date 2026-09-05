import pytest
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.vis.viewer import Visualizer3D


def test_visualizer_headless():
    try:
        ti.init(arch=ti.cpu)
    except RuntimeError:
        pytest.skip("CPU not available")

    domain = Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[1.0, 1.0, 1.0])
    atom = AtomSystem(max_atoms=10)
    atom.add_particles(
        x=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
        radius=[0.05, 0.05],
        density=2000.0,
    )

    try:
        vis = Visualizer3D(domain=domain, max_particles=10, show_window=False)
        res = vis.render_frame(atom=atom)
        assert isinstance(res, bool)
    except (RuntimeError, ValueError):
        pytest.skip("GGUI display not initialized in CI/headless environment")
