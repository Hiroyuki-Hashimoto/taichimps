"""
Example: Granular column collapse under gravity onto a flat bottom wall.
"""

import taichi as ti

from taichimps import AtomSystem, Computes, Domain, DumpWriter, NeighborList, Simulation
from taichimps.contact_history import ContactHistory
from taichimps.EXTRA_FIX import FixGravity, FixNVESphere, FixWallGran
from taichimps.GRANULAR import GranHookeHistory


def run_column_collapse(steps: int = 500) -> None:
    # Initialize Taichi
    ti.init(arch=ti.cpu)

    # 1. Create Domain (box [0, 1] x [0, 1] x [0, 1])
    domain = Domain(
        boxlo=[0.0, 0.0, 0.0],
        boxhi=[1.0, 1.0, 1.0],
        boundary=['p', 'p', 'f'],  # periodic in x, y; fixed in z
    )

    # 2. Create AtomSystem
    max_atoms = 500
    atom = AtomSystem(max_atoms=max_atoms)

    # Generate a small column of particles
    coords = []
    radii = []
    r = 0.02
    for ix in range(3):
        for iy in range(3):
            for iz in range(10):
                x = 0.45 + ix * (2.1 * r)
                y = 0.45 + iy * (2.1 * r)
                z = 0.05 + iz * (2.1 * r)
                coords.append([x, y, z])
                radii.append(r)

    atom.add_particles(
        x=coords,
        radius=radii,
        density=2500.0,  # 2500 kg/m^3 (sand / glass)
    )

    # 3. Neighbor list and history
    nlist = NeighborList(domain=domain, max_atoms=max_atoms, max_neighbors=64, skin=0.01)
    history = ContactHistory(max_atoms=max_atoms, max_neighbors=64)

    # 4. Granular force model (Hooke with shear history)
    pair = GranHookeHistory(
        domain=domain,
        kn=1.0e5,
        gamman=50.0,
        kt=2.0e4,
        gammat=20.0,
        xmu=0.5,
    )

    # 5. Simulation setup
    dt = 1e-4
    sim = Simulation(
        domain=domain,
        atom=atom,
        neighbor=nlist,
        history=history,
        pair=pair,
        dt=dt,
        thermo_freq=100,
    )

    # Add Fixes
    sim.add_fix(FixNVESphere(domain=domain))
    sim.add_fix(FixGravity(domain=domain, magnitude=9.81, direction=[0.0, 0.0, -1.0]))
    # Bottom wall at z = 0.0
    sim.add_fix(
        FixWallGran(
            domain=domain,
            wall_axis=2,
            wall_side=-1,
            wall_coord=0.0,
            kn=2.0e5,
            gamman=100.0,
            kt=5.0e4,
            gammat=50.0,
            xmu=0.5,
        )
    )

    # Add dump writer
    writer = DumpWriter(filepath="column_collapse.dump")
    sim.add_dump(writer, freq=100)

    print(f'Starting column collapse simulation with {atom.nlocal} particles...')
    sim.run(steps=steps)

    computes = Computes()
    print(f'Done! Final KE_trans={computes.ke_trans(atom):.6e}, KE_rot={computes.ke_rot(atom):.6e}')


if __name__ == '__main__':
    run_column_collapse()
