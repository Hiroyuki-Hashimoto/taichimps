"""
Real-time 3D interactive demonstration of granular column collapse using Taichi GGUI.
"""

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.gravity import FixGravity
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.EXTRA_FIX.wall_gran import FixWallGran
from taichimps.GRANULAR.hooke_history import GranHookeHistory
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation
from taichimps.vis import Visualizer3D


def run_column_collapse_interactive(show_window: bool = True, max_steps: int = 2000) -> None:
    ti.init(arch=ti.vulkan)

    boxlo = [-0.1, -0.1, 0.0]
    boxhi = [0.5, 0.5, 0.6]
    domain = Domain(boxlo=boxlo, boxhi=boxhi, boundary=("f", "f", "f"))

    max_particles = 1000
    atom = AtomSystem(max_atoms=max_particles)

    # Initialize a column of particles
    r = 0.008
    spacing = 2.1 * r
    nx, ny, nz = 6, 6, 16
    x_coords = []
    radii = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                px = 0.05 + i * spacing + (np.random.rand() - 0.5) * 0.001
                py = 0.05 + j * spacing + (np.random.rand() - 0.5) * 0.001
                pz = 0.02 + k * spacing
                x_coords.append([px, py, pz])
                radii.append(r)

    atom.add_particles(x=x_coords, radius=radii, density=2500.0)

    # Setup Simulation components
    nlist = NeighborList(domain=domain, max_atoms=max_particles, skin=0.002)
    history = ContactHistory(max_atoms=max_particles)
    pair = GranHookeHistory(
        domain=domain,
        kn=2e4,
        gamman=5.0,
        kt=1e4,
        gammat=5.0,
        xmu=0.5,
    )

    sim = Simulation(
        domain=domain,
        atom=atom,
        neighbor=nlist,
        history=history,
        pair=pair,
        dt=1e-4,
    )

    sim.add_fix(FixNVESphere(domain=domain))
    sim.add_fix(FixGravity(domain=domain, magnitude=9.81, direction=[0.0, 0.0, -1.0]))
    # Bottom floor
    sim.add_fix(
        FixWallGran(
            domain=domain,
            wall_axis=2,
            wall_side=-1,
            wall_coord=0.0,
            kn=5e4,
            gamman=10.0,
            kt=2e4,
            gammat=10.0,
            xmu=0.5,
        )
    )

    sim.init_simulation()

    # Create Real-time Visualizer
    vis = Visualizer3D(
        domain=domain,
        max_particles=max_particles,
        title="taichimps: Real-time Granular Column Collapse (GGUI)",
        show_window=show_window,
    )

    step = 0
    substeps_per_frame = 10
    print(f"Running interactive simulation ({atom.nlocal} particles)... Press Space to pause, RMB to rotate.")

    while step < max_steps:
        if not vis.paused:
            for _ in range(substeps_per_frame):
                sim.step()
                step += 1

        is_running = vis.render_frame(
            atom=atom,
            particle_radius=r,
            color_by="speed",
            max_speed=0.5,
        )
        if not is_running:
            break

    print(f"Simulation finished at step {step}.")


if __name__ == "__main__":
    run_column_collapse_interactive(show_window=True)
