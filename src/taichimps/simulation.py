"""
Main Simulation Orchestrator for taichimps.
Reference: LAMMPS src/lammps.cpp, src/run.cpp
"""

from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.computes.thermo import Computes
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.dump import DumpWriter
from taichimps.EXTRA_FIX.base import Fix
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.GRANULAR.base import GranularPair
from taichimps.neighbor import NeighborList


@ti.data_oriented
class Simulation:
    """
    Coordinates the time integration cycle, neighbor rebuilds, force computation, and output.
    """

    def __init__(
        self,
        domain: Domain,
        atom: AtomSystem | None = None,
        neighbor: NeighborList | None = None,
        history: ContactHistory | None = None,
        pair: GranularPair | None = None,
        max_atoms: int = 100000,
        max_neighbors_per_atom: int = 64,
        skin: float = 0.001,
        dt: float = 1e-4,
        thermo_freq: int = 100,
        float_type: Any = ti.f64,
    ) -> None:
        self.domain = domain
        self.dt = dt
        self.float_type = float_type
        self.timestep = 0

        self.atom = atom if atom is not None else AtomSystem(max_atoms=max_atoms, float_type=float_type)
        self.neighbor = (
            neighbor
            if neighbor is not None
            else NeighborList(
                domain=domain,
                max_atoms=max_atoms,
                max_neighbors_per_atom=max_neighbors_per_atom,
                skin=skin,
                float_type=float_type,
            )
        )
        self.history = (
            history
            if history is not None
            else ContactHistory(
                max_atoms=max_atoms,
                max_neighbors=max_neighbors_per_atom,
                float_type=float_type,
            )
        )

        self.pair_style: GranularPair | None = pair
        self.fixes: list[Fix] = []
        self.dumps: list[tuple[DumpWriter, int]] = []
        self.thermo_freq = thermo_freq
        self.computes = Computes(float_type=float_type)

    def add_dump(self, dump: DumpWriter, freq: int = 100) -> None:
        """Add a dump writer at the specified frequency."""
        self.dumps.append((dump, freq))

    def set_pair_style(self, pair: GranularPair) -> None:
        """Set the granular contact pair style."""
        self.pair_style = pair

    def add_fix(self, fix: Fix) -> None:
        """Add a fix (e.g. FixGravity, FixWallGran, FixNVESphere)."""
        if isinstance(fix, FixNVESphere):
            self.integrator = fix
        else:
            self.fixes.append(fix)

    def init_simulation(self) -> None:
        """Initialize forces for the first timestep if not already done."""
        # Calculate initial forces if needed
        self.atom.clear_forces()
        if self.pair_style is not None:
            self.neighbor.check_and_build(self.atom, self.timestep)
            self.pair_style.compute(
                self.atom,
                self.neighbor,
                self.history,
                self.dt,
            )
        for fix in self.fixes:
            fix.post_force(self.atom, self.dt)

    def step(self) -> None:
        """
        Execute a single Velocity Verlet DEM timestep:
        1. Fix initial_integrate (v half-step, x full-step, omega half-step)
        2. Check neighbor build
        3. Clear forces & torques
        4. Compute pair forces (Granular contact)
        5. Fix post_force (Gravity, Wall contact)
        6. Fix final_integrate (v 2nd half-step, omega 2nd half-step)
        """
        # Step 1: Initial integration
        if self.integrator is not None:
            self.integrator.initial_integrate(self.atom, self.dt)

        # Step 2: Neighbor list build check
        rebuilt = self.neighbor.check_and_build(self.atom, self.timestep)
        if rebuilt and self.history is not None:
            self.history.compress_and_update(self.neighbor)

        # Step 3: Clear forces
        self.atom.clear_forces()

        # Step 4: Pair force computation
        if self.pair_style is not None:
            self.pair_style.compute(
                self.atom,
                self.neighbor,
                self.history,
                self.dt,
            )

        # Step 5: Fix post_force (e.g. Gravity, Walls)
        for fix in self.fixes:
            fix.post_force(self.atom, self.dt)

        # Step 6: Final integration
        if self.integrator is not None:
            self.integrator.final_integrate(self.atom, self.dt)

        # Step 7: Fix end_of_step (e.g. deform/pressure)
        for fix in self.fixes:
            fix.end_of_step(self.atom, self.dt)

        self.timestep += 1

        # Periodic dump output
        for dump_writer, freq in self.dumps:
            if self.timestep % freq == 0:
                dump_writer.write_dump(self.timestep, self.domain, self.atom)

    def run(self, steps: int) -> None:
        """Run the simulation for a given number of steps."""
        if self.timestep == 0:
            self.init_simulation()
        for _ in range(steps):
            self.step()
