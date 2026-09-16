import tempfile
from pathlib import Path

import pytest
import taichi as ti

from taichimps.input import LAMMPSInputParser


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def test_parser_extra_taichi_commands():
    script = """
atom_style sphere
boundary f f f
region box block 0 20 0 20 0 20
create_box 1 box

pair_style gran/hooke 1e5 1e4 10.0 5.0 0.5 0

create_atoms 1 single 10 10 2
set group all density 1000.0 diameter 1.0

fix 1 all nve/sphere
fix 2 all wave z ricker f0 10.0 amp 50.0 mode force
fix 3 all sponge thick 3.0 eta 20.0 power 2.0
fix 4 all probe 1 10.0 10.0 10.0 radius 2.0 every 1

timestep 0.001
run 5
"""
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        parser = LAMMPSInputParser(Path(script_path), default_fp=ti.f64, arch=ti.cpu)
        parser.execute()

        sim = parser.simulation
        assert sim is not None
        assert sim.timestep == 5

        # Check that fixes were instantiated and added
        fix_names = [f.__class__.__name__ for f in sim.fixes]
        assert "FixWave" in fix_names
        assert "WinSponge" in fix_names
        assert "FixProbe" in fix_names

        probe_fix = next(f for f in sim.fixes if f.__class__.__name__ == "FixProbe")
        print(f"DEBUG: fixes in parser: {list(parser.fixes.keys())}")
        print(f"DEBUG: fixes in sim: {[f.__class__.__name__ for f in sim.fixes]}")
        print(f"DEBUG: probe_fix instance is in sim.fixes: {probe_fix in sim.fixes}")
        print(f"DEBUG: probe_fix.step_count: {probe_fix.step_count}, nevery: {probe_fix.nevery}")
        print(f"DEBUG: probe_fix.data_history len: {len(probe_fix.data_history)}")
        print(f"DEBUG: probe_fix.n_probes: {probe_fix.n_probes}, probe_coords: {probe_fix.points_np}")
        data = probe_fix.to_numpy()
        assert data.shape[0] == 5  # 5 steps recorded
        assert data.shape[1] == 1  # 1 probe
        assert data.shape[2] == 3  # 3 fields (vx, vy, vz)

    finally:
        Path(script_path).unlink(missing_ok=True)
