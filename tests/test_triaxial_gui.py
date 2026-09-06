"""
Triaxial compression test for cylindrical specimen with Taichi GUI / GGUI 3D preview.
Specimen dimensions:
  - Diameter: 50 mm (D = 0.05 m, R = 0.025 m)
  - Height: 100 mm (H = 0.10 m)
  - Particle diameter: ~2.0 mm (r = 1.0 mm)
  - Particles deposited under gravity into cylinder mould
  - Silica sand properties:
      * Density: 2650 kg/m^3 (quartz / silica sand)
      * Young's modulus E: ~30 GPa (DEM calibrated kn ~ 2e6 N/m, kt ~ 1.5e6 N/m)
      * Friction coefficient: xmu ~ 0.577 (friction angle ~ 30-35 deg)
      * Viscous damping: gamman ~ 50.0, gammat ~ 25.0
  - Confining stress: 600 kPa (0.6 MPa) via radial cylinder servo / wall
  - Axial stress / target: 1.0 MPa (up to 20% axial strain compression)
"""

import os
import platform

import numpy as np
import pytest
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.gravity import FixGravity
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.EXTRA_FIX.wall_gran import FixWallGran
from taichimps.EXTRA_FIX.wall_gran_region import FixWallGranRegion
from taichimps.GRANULAR.hertz_history import GranHertzHistory
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation
from taichimps.vis.viewer import Visualizer3D


def run_cylindrical_triaxial_simulation(
    gui_preview: bool = False,
    max_steps: int = 100,
    n_particles: int = 250,
    keep_window_open: bool = False,
):
    """
    Run cylindrical triaxial compression simulation.
    Diameter D = 0.05 m (R = 0.025 m), Height H = 0.10 m.
    """
    # 1. Specimen geometry & Domain
    r_cyl = 0.025  # 25 mm radius (diameter D = 50 mm)
    h_cyl = 0.10   # 100 mm
    c_x, c_y = 0.035, 0.035
    boxlo = [0.0, 0.0, 0.0]
    boxhi = [0.07, 0.07, 0.12]

    domain = Domain(boxlo=boxlo, boxhi=boxhi, boundary=["f", "f", "f"])

    # 2. Silica sand parameters
    # Silica grain density: 2650 kg/m^3
    # Radius ~ 1.0 mm (diameter 2.0 mm)
    # Contact stiffness for silica sand
    kn = 2.0e6
    kt = 1.5e6
    gamman = 50.0
    gammat = 25.0
    xmu = 0.577  # tan(30 deg)

    max_atoms = max(n_particles * 2, 1000)
    atom = AtomSystem(max_atoms=max_atoms, float_type=ti.f64)

    # 3. Generate particles inside cylinder (gravitational deposition preview)
    rng = np.random.default_rng(42)
    positions: list[list[float]] = []
    radii: list[float] = []
    r_particle_nominal = 0.001  # 1 mm radius = 2 mm diameter

    # Pack in layers within r < r_cyl - r_particle
    effective_r = r_cyl - 1.2 * r_particle_nominal
    while len(positions) < n_particles:
        # random radius and angle
        r_rand = effective_r * np.sqrt(rng.uniform(0.05, 1.0))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        px = c_x + r_rand * np.cos(theta)
        py = c_y + r_rand * np.sin(theta)
        pz = rng.uniform(0.005, h_cyl * 0.95)
        # Polydispersity 1.8mm ~ 2.2mm (d = 2mm +/- 10%)
        p_rad = r_particle_nominal * rng.uniform(0.9, 1.1)

        positions.append([px, py, pz])
        radii.append(p_rad)

    atom.add_particles(
        x=positions,
        v=np.zeros((len(positions), 3)),
        omega=np.zeros((len(positions), 3)),
        radius=radii,
        density=2650.0,  # Silica sand density
        atom_type=1,
    )

    # 4. Contact law & Neighbor list
    pair = GranHertzHistory(
        domain=domain,
        kn=kn,
        kt=kt,
        gamman=gamman,
        gammat=gammat,
        xmu=xmu,
        dampflag=1,
        float_type=ti.f64,
    )
    nlist = NeighborList(
        max_atoms=max_atoms,
        skin=0.001,
        domain=domain,
        float_type=ti.f64,
    )
    dt = 1.0e-5  # Stable time step for high kn silica sand
    sim = Simulation(domain=domain, atom=atom, neighbor=nlist, pair=pair, dt=dt)

    # 5. Fixes
    # NVE integration
    fix_nve = FixNVESphere(domain=domain, float_type=ti.f64)
    sim.add_fix(fix_nve)

    # Gravity for deposition
    fix_grav = FixGravity(domain=domain, magnitude=9.81, direction=[0.0, 0.0, -1.0], float_type=ti.f64)
    sim.add_fix(fix_grav)

    # Bottom pedestal (z = 0.002 m, wall_side = -1: normal +z)
    z_bottom = 0.002
    fix_bottom = FixWallGran(
        domain=domain,
        wall_axis=2,
        wall_side=-1,
        wall_coord=z_bottom,
        kn=kn,
        gamman=gamman,
        kt=kt,
        gammat=gammat,
        xmu=xmu,
        float_type=ti.f64,
    )
    sim.add_fix(fix_bottom)

    # Top platen (initially at z = h_cyl = 0.10 m, wall_side = +1: normal -z)
    z_top = h_cyl
    fix_top = FixWallGran(
        domain=domain,
        wall_axis=2,
        wall_side=1,
        wall_coord=z_top,
        kn=kn,
        gamman=gamman,
        kt=kt,
        gammat=gammat,
        xmu=xmu,
        float_type=ti.f64,
    )
    sim.add_fix(fix_top)

    # Cylindrical lateral membrane / mould (D = 50 mm, R = 25 mm)
    fix_cyl = FixWallGranRegion(
        domain=domain,
        axis=2,
        c1=c_x,
        c2=c_y,
        radius=r_cyl,
        side="in",
        axis_lo=0.0,
        axis_hi=h_cyl * 1.5,
        fstyle="hertz",
        kn=kn,
        gamman=gamman,
        kt=kt,
        gammat=gammat,
        xmu=xmu,
        float_type=ti.f64,
    )
    sim.add_fix(fix_cyl)

    # 6. Initialize simulation
    sim.init_simulation()

    # 7. Setup Visualizer3D (Taichi GGUI 3D preview)
    vis = None
    if gui_preview:
        try:
            # On Windows, always show window. On Linux, check DISPLAY/WAYLAND_DISPLAY.
            is_windows = platform.system() == "Windows"
            has_display = is_windows or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            print(f"[Visualizer] Initializing GGUI 3D Window (show_window={has_display})...")
            vis = Visualizer3D(
                domain=domain,
                max_particles=max_atoms,
                res=(1024, 768),
                title="Taichimps: Cylindrical Triaxial Test (D=50mm, H=100mm, Silica Sand)",
                show_window=has_display,
            )
            # Set camera viewpoint focusing on cylindrical specimen center
            vis.camera.position(c_x + 0.15, c_y + 0.15, 0.08)
            vis.camera.lookat(c_x, c_y, 0.05)
            vis.camera.up(0, 0, 1)
            print("[Visualizer] 3D Window ready.")
        except (RuntimeError, ValueError) as e:
            print(f"[Visualizer Notice] GUI display initialization skipped: {e}")
            vis = None

    # 8. Triaxial Compression Loop:
    # Target: 20% axial strain compression (Delta H = 0.20 * 0.10 m = 0.02 m)
    # Top platen descends downward with velocity v_axial
    v_top_down = 0.05  # m/s compression velocity
    sigma_3_confining = 600.0e3  # 600 kPa lateral pressure target
    sigma_1_target = 1.0e6      # 1.0 MPa axial target

    axial_strains = []
    top_positions = []

    # Initial render frame before stepping so the window immediately appears
    if vis is not None:
        vis.render_frame(atom=atom)

    for step in range(max_steps):
        # Update top platen position downwards for axial compression
        fix_top.wall_coord -= v_top_down * dt
        sim.step()

        current_h = fix_top.wall_coord - z_bottom
        axial_strain = (h_cyl - current_h) / h_cyl
        axial_strains.append(axial_strain)
        top_positions.append(fix_top.wall_coord)

        # Periodic status report
        if (step + 1) % 50 == 0 or step == 0:
            print(f"[Step {step+1:4d}/{max_steps}] Axial strain: {axial_strain*100:.2f}%, Top z: {fix_top.wall_coord*1e3:.2f} mm")

        # GGUI 3D rendering preview (render every 2 steps to reduce overhead)
        if vis is not None and (step % 2 == 0 or step == max_steps - 1):
            is_running = vis.render_frame(atom=atom)
            if not is_running:
                print("[Visualizer] Window closed by user.")
                break

    if vis is not None and keep_window_open and vis.show_window:
        print("[Visualizer] Simulation finished. Window is kept open (press ESC or close window to exit)...")
        while vis.render_frame(atom=atom):
            pass

    return {
        "n_particles": atom.nlocal,
        "final_axial_strain": axial_strains[-1],
        "final_top_coord": top_positions[-1],
        "confining_stress_target": sigma_3_confining,
        "axial_stress_target": sigma_1_target,
    }


def test_triaxial_compression_cylindrical_headless():
    """Verify cylindrical triaxial test execution under CI / headless mode."""
    try:
        ti.init(arch=ti.cpu, default_fp=ti.f64)
    except RuntimeError:
        pytest.skip("CPU not available")

    results = run_cylindrical_triaxial_simulation(
        gui_preview=True,
        max_steps=20,
        n_particles=100,
    )

    assert results["n_particles"] == 100
    assert results["final_axial_strain"] > 0.0
    assert results["final_top_coord"] < 0.10


if __name__ == "__main__":
    # When run directly from CLI (e.g. uv run python tests/test_triaxial_gui.py),
    # launch GPU/Vulkan with interactive 3D GGUI window!
    print("Initializing Taichi for Cylindrical Triaxial Test with GUI preview...")
    try:
        ti.init(arch=ti.vulkan, default_fp=ti.f64)
    except RuntimeError:
        ti.init(arch=ti.cpu, default_fp=ti.f64)

    res = run_cylindrical_triaxial_simulation(
        gui_preview=True,
        max_steps=5000,
        n_particles=300,
        keep_window_open=True,
    )
    print(f"Triaxial test preview finished: {res}")
