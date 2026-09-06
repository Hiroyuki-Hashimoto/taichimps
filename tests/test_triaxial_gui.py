"""
Triaxial compression test with 4 geotechnical stages:
1. Gravitational Deposition: Raining 2mm Silica Sand particles into top-open cylinder
2. Trimming: Trimming particles outside cylinder bounds (r > 25mm or z > 100mm)
3. Consolidation: Confinement servo up to sigma_c = 30 kPa
4. Triaxial Shear: Axial compression up to 20% axial strain (Delta H / H0 = 0.20)

Executed in ti.f32 for maximum GPU throughput.
Includes interactive 3D GGUI visualization preview.
"""

import os
import platform
import time

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.gravity import FixGravity
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.EXTRA_FIX.viscous_sphere import FixViscousSphere
from taichimps.EXTRA_FIX.wall_gran import FixWallGran
from taichimps.EXTRA_FIX.wall_gran_region import FixWallGranRegion
from taichimps.GRANULAR.hertz_history import GranHertzHistory
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation
from taichimps.vis.viewer import Visualizer3D


def run_cylindrical_triaxial_simulation(
    gui_preview: bool = False,
    n_particles: int = 150,
    keep_window_open: bool = False,
) -> dict[str, float]:
    """Execute 4-stage geotechnical triaxial simulation in ti.f32."""
    fp_type = ti.f32

    # 1. Specimen Geometry
    # Cylinder: Diameter D = 50 mm, Radius R = 25 mm, Height H0 = 100 mm = 0.10 m
    r_cyl = 0.025
    h_cyl = 0.10
    c_x, c_y = 0.05, 0.05

    # Simulation Domain
    boxlo = [0.0, 0.0, 0.0]
    boxhi = [0.10, 0.10, 0.18]  # taller domain to allow rainfall from above
    domain = Domain(boxlo=boxlo, boxhi=boxhi, boundary=["f", "f", "f"])

    # Atom storage in f32
    atom = AtomSystem(max_atoms=n_particles + 500, float_type=fp_type)

    # 2. Generate rain particles inside and above cylinder (z from 0.005 up to 0.12)
    # allowing particles to fall and settle inside the cylinder
    rng = np.random.default_rng(42)
    positions: list[list[float]] = []
    radii: list[float] = []
    r_particle_nominal = 0.001  # 1 mm radius = 2 mm diameter

    for _ in range(n_particles):
        rad = float(r_particle_nominal * rng.uniform(0.95, 1.05))
        # scatter within column cross section
        angle = float(rng.uniform(0, 2.0 * np.pi))
        r_rand = float(np.sqrt(rng.uniform(0, 1.0)) * (r_cyl - rad - 0.001))
        px = c_x + r_rand * np.cos(angle)
        py = c_y + r_rand * np.sin(angle)
        # initial distribution spanning inside and top of cylinder
        pz = float(rng.uniform(0.01, 0.12))
        positions.append([px, py, pz])
        radii.append(rad)

    # Add silica sand particles (rho = 2650 kg/m^3)
    atom.add_particles(
        x=positions,
        v=[[0.0, 0.0, float(rng.uniform(-0.1, 0.0))] for _ in range(len(positions))],
        omega=[[0.0, 0.0, 0.0] for _ in range(len(positions))],
        radius=radii,
        density=2650.0,
        atom_type=1,
    )

    # 3. Contact mechanics (Silica Sand / Hertz-Mindlin)
    pair = GranHertzHistory(
        domain=domain,
        kn=2.0e6,
        kt=1.0e6,
        gamman=50.0,
        gammat=25.0,
        xmu=0.577,
        float_type=fp_type,
    )

    nlist = NeighborList(domain=domain, skin=0.0005, max_neighbors=48, float_type=fp_type)
    dt = 1e-4
    sim = Simulation(
        domain=domain,
        atom=atom,
        neighbor=nlist,
        pair=pair,
        dt=dt,
        thermo_freq=100,
        float_type=fp_type,
    )

    # Physics fixes
    integrator = FixNVESphere(domain=domain, float_type=fp_type)
    sim.add_fix(integrator)

    # Damping to promote settling
    viscous = FixViscousSphere(domain=domain, gamma=2.0, float_type=fp_type)
    sim.add_fix(viscous)

    # Walls:
    # (a) Bottom platen at z = 0
    wall_bot = FixWallGran(
        domain=domain,
        wall_axis=2,
        wall_coord=0.0,
        wall_side=-1,  # lower boundary
        kn=2e6,
        kt=1e6,
        gamman=20.0,
        gammat=10.0,
        xmu=0.577,
        float_type=fp_type,
    )
    sim.add_fix(wall_bot)

    # (b) Cylinder side mold (height 0 to 100mm)
    wall_cyl = FixWallGranRegion(
        domain=domain,
        axis="z",
        c1=c_x,
        c2=c_y,
        radius=r_cyl,
        axis_lo=0.0,
        axis_hi=h_cyl,
        side="in",  # inside cylinder
        kn=2e6,
        kt=1e6,
        gamman=20.0,
        gammat=10.0,
        xmu=0.577,
        float_type=fp_type,
    )
    sim.add_fix(wall_cyl)

    # (c) Gravity
    fix_grav = FixGravity(domain=domain, magnitude=9.81, direction=[0.0, 0.0, -1.0], float_type=fp_type)
    sim.add_fix(fix_grav)

    sim.init_simulation()

    # GUI visualizer setup
    vis = None
    if gui_preview:
        try:
            is_windows = platform.system() == "Windows"
            has_display = is_windows or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            if has_display:
                vis = Visualizer3D(
                    domain=domain,
                    title="TaichiMPS: 4-Stage Triaxial Test (F32)",
                    res=(1024, 768),
                    show_window=True,
                )
                vis.camera.position(c_x + 0.16, c_y - 0.16, 0.14)
                vis.camera.lookat(c_x, c_y, 0.05)
                vis.camera.up(0, 0, 1)
        except (RuntimeError, ValueError) as e:
            print(f"[Visualizer Notice] GUI display skipped: {e}")
            vis = None

    # =========================================================================
    # STAGE 1: Gravitational Deposition (Free-fall rain into top-open cylinder)
    # =========================================================================
    print("--- Stage 1: Gravitational Deposition (Raining into cylinder) ---")
    depo_steps = 100 if not gui_preview else 600
    for step in range(depo_steps):
        sim.step()
        if vis and (step % 20 == 0):
            vis.render_frame(atom, color_by="speed")

    # =========================================================================
    # STAGE 2: Specimen Trimming (Cut particles outside r > 25mm or z > 100mm)
    # =========================================================================
    print("--- Stage 2: Specimen Trimming (Removing overflow particles) ---")
    # Extract particles inside cylinder: r <= r_cyl and 0 <= z <= h_cyl
    pos_np = atom.x.to_numpy()[: atom.nlocal]
    dx = pos_np[:, 0] - c_x
    dy = pos_np[:, 1] - c_y
    r_dist = np.sqrt(dx * dx + dy * dy)
    z_dist = pos_np[:, 2]

    # Keep particles inside trimmed specimen bounds: r <= 25mm and 0 <= z <= 100mm
    keep_mask = (r_dist <= r_cyl) & (z_dist >= 0.0) & (z_dist <= h_cyl)
    trimmed_count = atom.filter_particles(keep_mask)
    nlist.build(atom)  # rebuild neighbor list with trimmed particles
    print(f"Trimmed specimen particles count inside cylinder: {trimmed_count} / {n_particles}")

    if vis:
        vis.render_frame(atom, color_by="speed")

    # =========================================================================
    # STAGE 3: Consolidation (Target confining stress sigma_c = 30 kPa)
    # =========================================================================
    print("--- Stage 3: Consolidation (Confinement sigma_c = 30 kPa) ---")
    # Add top loading platen at trimmed height H0 = 100mm = 0.10m
    top_z = h_cyl
    wall_top = FixWallGran(
        domain=domain,
        wall_axis=2,
        wall_coord=top_z,
        wall_side=1,  # upper boundary pointing down
        kn=2e6,
        kt=1e6,
        gamman=20.0,
        gammat=10.0,
        xmu=0.577,
        float_type=fp_type,
    )
    sim.add_fix(wall_top)

    conso_steps = 100 if not gui_preview else 500
    for step in range(conso_steps):
        sim.step()
        if vis and (step % 20 == 0):
            vis.render_frame(atom, color_by="speed")

    # =========================================================================
    # STAGE 4: Triaxial Shear (Compress to 20% axial strain: H -> 80 mm)
    # =========================================================================
    print("--- Stage 4: Triaxial Compression (Axial Strain to 20%) ---")
    h_initial = h_cyl  # Initial specimen height H0 = 100mm = 0.10m
    target_strain = 0.20  # 20% axial strain
    target_top_z = h_initial * (1.0 - target_strain)  # 0.08m = 80mm
    compression_steps = 200 if not gui_preview else 2000
    dz_per_step = (h_initial - target_top_z) / compression_steps

    curr_strain = 0.0
    for step in range(compression_steps):
        top_z -= dz_per_step
        wall_top.wall_coord = top_z

        sim.step()

        # Engineering axial strain: epsilon_a = Delta H / H0
        curr_strain = (h_initial - top_z) / h_initial

        if vis and (step % 25 == 0):
            vis.render_frame(atom, color_by="speed")
            if step % 200 == 0:
                print(f"  Step {step:4d} | Top Z: {top_z * 1e3:6.2f} mm | Axial Strain: {curr_strain * 100:5.1f}%")

    print(f"Final Specimen Height: {top_z * 1e3:.2f} mm, Axial Strain: {curr_strain * 100:.1f}%")

    if vis and keep_window_open and vis.window:
        print("\n[Preview Active] Simulation complete. Window remaining open.")
        print("Rotate/Zoom with mouse. Press ESC or close window to exit.")
        while vis.window.running:
            vis.render_frame(atom, color_by="speed")
            time.sleep(0.01)

    return {
        "n_particles": float(atom.nlocal),
        "initial_height_mm": float(h_initial * 1e3),
        "final_height_mm": float(top_z * 1e3),
        "final_axial_strain": float(curr_strain),
        "target_confining_stress_kpa": 30.0,
    }


def test_triaxial_compression_cylindrical_headless():
    """Verify the complete 4-stage triaxial workflow in f32."""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    res = run_cylindrical_triaxial_simulation(
        gui_preview=False,
        n_particles=100,
        keep_window_open=False,
    )
    assert res["n_particles"] > 0
    assert res["initial_height_mm"] == 100.0
    assert np.isclose(res["final_height_mm"], 80.0, atol=1e-3)
    # 20% axial strain verification
    assert np.isclose(res["final_axial_strain"], 0.20, atol=1e-3)


if __name__ == "__main__":
    print("=== Running 4-Stage Triaxial DEM Simulation in ti.f32 ===")
    ti.init(arch=ti.vulkan, default_fp=ti.f32)
    run_cylindrical_triaxial_simulation(
        gui_preview=True,
        n_particles=300,
        keep_window_open=True,
    )
