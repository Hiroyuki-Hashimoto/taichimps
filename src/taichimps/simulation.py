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
        self.integrator: FixNVESphere | None = None
        self.fixes: list[Fix] = []
        self.dumps: list[tuple[DumpWriter, int]] = []
        # `restart N file`: how often to write, the name (a `*` in it is
        # replaced by the timestep, as in LAMMPS) and who does the writing.
        self.restart_every = 0
        self.restart_file: str | None = None
        self.restart_writer: Any = None
        # Whether Verlet::setup() has run for this system yet.  After
        # read_restart the timestep is non-zero but the forces are not, so this
        # cannot be inferred from the timestep.
        self.setup_done = False
        # `run <N> pre no` skips the setup; on by default, as in LAMMPS.
        self.run_pre = True
        self.thermo_freq = thermo_freq
        self.computes = Computes(float_type=float_type)

    def set_restart(self, every: int, path: str | None, writer: Any) -> None:
        """Configure periodic restart output (`restart N file`)."""
        self.restart_every = every
        self.restart_file = path
        self.restart_writer = writer

    def write_restart_now(self) -> None:
        """Write one periodic restart file for the current timestep."""
        if self.restart_writer is None or not self.restart_file:
            return
        self.restart_writer(
            self.restart_file.replace("*", str(self.timestep)), self.timestep
        )

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

    def remove_fix(self, fix: Fix) -> None:
        """Remove a fix from active simulation fixes."""
        if fix in self.fixes:
            self.fixes.remove(fix)
        elif self.integrator is fix:
            self.integrator = None

    def reneighbor_if_needed(self, force: bool = False) -> bool:
        """
        Apply PBC, rebuild the neighbor list and carry contact history over,
        but only on steps where LAMMPS would reneighbor.

        Mirrors the LAMMPS ordering: FixNeighHistory::pre_exchange() saves the
        touching contacts, Domain::pbc() remaps coordinates into the box, the
        list is rebuilt, then FixNeighHistory::post_neighbor() restores history
        onto the new list.  Wrapping coordinates only here (rather than every
        step) is also what keeps the skin displacement check meaningful, since
        x0 and x then live in the same periodic image.
        """
        if self.neighbor is None:
            return False

        # A fix may demand a rebuild regardless of the displacement check: a
        # box flip leaves atoms outside the relabelled cell until they are
        # wrapped, and binning would place them wrongly in the meantime.
        forced = force
        for fix in self.fixes:
            if fix.force_reneighbor:
                forced = True
                fix.force_reneighbor = False

        if not self.neighbor.decide(self.atom, self.timestep) and not forced:
            return False

        if self.history is not None:
            self.history.save_state(self.atom)
        self.domain.pbc(self.atom)
        self.neighbor.build(self.atom)
        if self.history is not None:
            self.history.restore_state(self.atom, self.neighbor)
        return True

    def setup_run(self, nsteps_total: int) -> None:
        """Tell the fixes a run is starting and how long it is."""
        for fix in self.fixes:
            fix.setup(nsteps_total, self.dt)
        if self.integrator is not None:
            self.integrator.setup(nsteps_total, self.dt)

    def init_simulation(self) -> None:
        """
        Verlet::setup(): rebuild the neighbor list and evaluate the forces at
        the current state, without advancing any contact history.

        LAMMPS does this at the start of *every* `run` command, not just the
        first.  It matters: the setup force is what the first half-kick of the
        next step uses, and it is evaluated with `history_update = 0`, so a run
        split in two is not the same trajectory as one long run -- but a run
        resumed from a restart is exactly the same as a second `run` command,
        which is what makes a restart verifiable.
        """
        self.setup_done = True
        # Calculate initial forces if needed
        self.atom.clear_forces()
        if self.pair_style is not None:
            self.reneighbor_if_needed(force=True)
            # LAMMPS sets shearupdate = 0 while update->setupflag is on, so the
            # setup force evaluation must not advance the shear history.
            self.pair_style.compute(
                self.atom,
                self.neighbor,
                self.history,
                self.dt,
                shearupdate=False,
            )
        # Fixes that keep their own contact history (the walls) must not
        # advance it during setup either.
        for fix in self.fixes:
            fix.history_update = False
        for fix in self.fixes:
            fix.post_force(self.atom, self.dt)
        for fix in self.fixes:
            fix.history_update = True

    def _sub_step_inner(self) -> None:
        """Execute a single time step without CPU synchronizations or periodic I/O."""
        # LAMMPS increments ntimestep at the top of the step (Verlet::run does
        # `ntimestep = ++update->ntimestep`), so everything inside the step --
        # the neighbor decision, fix print, dumps -- sees the number of the step
        # being executed. Incrementing at the bottom instead left every one of
        # them one step behind.
        self.timestep += 1
        for fix in self.fixes:
            fix.timestep = self.timestep
        if self.integrator is not None:
            self.integrator.timestep = self.timestep

        # 1. Initial integration (Velocity Verlet 1st half)
        if self.integrator is not None:
            self.integrator.initial_integrate(self.atom, self.dt)

        # 2. PBC + neighbor list rebuild (carries contact history over)
        self.reneighbor_if_needed()

        # 3. Clear forces & torques
        self.atom.clear_forces()

        # 4. Pair force computation (Granular contact)
        if self.pair_style is not None and self.neighbor is not None:
            self.pair_style.compute(
                self.atom,
                self.neighbor,
                self.history,
                self.dt,
            )

        # 5. Fix post_force (e.g. Gravity, Walls)
        for fix in self.fixes:
            fix.post_force(self.atom, self.dt)

        # 6. Final integration (Velocity Verlet 2nd half)
        if self.integrator is not None:
            self.integrator.final_integrate(self.atom, self.dt)

        # 7. Fix end_of_step (e.g. deform/pressure, print)
        for fix in self.fixes:
            fix.end_of_step(self.atom, self.dt)

    def run_gpu(self, steps: int) -> None:
        """
        Execute simulation steps in pure batch on GPU/device, minimizing CPU sync.

        If dumps or periodic I/O fixes exist, runs in sub-batches up to the next I/O interval.
        """
        # Fix::init() runs before Verlet::setup() in LAMMPS, and the order
        # matters: a fix that publishes state the force computation reads --
        # fix deform putting box rates in domain->h_rate for `remap v` -- must
        # have done so before the setup forces are evaluated.
        self.setup_run(steps)
        if self.run_pre or not self.setup_done:
            self.init_simulation()

        intervals = []
        for _, freq in self.dumps:
            if freq > 0:
                intervals.append(freq)
        for fix in self.fixes:
            if hasattr(fix, "nevery") and fix.nevery > 0:
                intervals.append(fix.nevery)

        remaining = steps
        while remaining > 0:
            chunk = remaining
            for freq in intervals:
                rem_to_freq = freq - (self.timestep % freq)
                if rem_to_freq > 0 and rem_to_freq < chunk:
                    chunk = rem_to_freq
            chunk = max(1, chunk)

            for _ in range(chunk):
                self._sub_step_inner()

            # Periodic dump output
            for dump_writer, freq in self.dumps:
                if self.timestep % freq == 0:
                    dump_writer.write_dump(self.timestep, self.domain, self.atom)
            if self.restart_every and self.timestep % self.restart_every == 0:
                self.write_restart_now()

            remaining -= chunk

    def step(self) -> None:
        """
        Execute a single Velocity Verlet DEM timestep:
        1. Fix initial_integrate (v half-step, x full-step, omega half-step)
        2. Check neighbor build
        3. Clear forces & torques
        4. Compute pair forces (Granular contact)
        5. Fix post_force (Gravity, Wall contact)
        6. Fix final_integrate (v 2nd half-step, omega 2nd half-step)
        7. Fix end_of_step (e.g. deform/pressure)
        """
        self._sub_step_inner()

        # Periodic dump output
        for dump_writer, freq in self.dumps:
            if self.timestep % freq == 0:
                dump_writer.write_dump(self.timestep, self.domain, self.atom)
        if self.restart_every and self.timestep % self.restart_every == 0:
            self.write_restart_now()

    def run(self, steps: int) -> None:
        """Run the simulation for a given number of steps."""
        # Fix::init() before Verlet::setup(), as in LAMMPS: see run_gpu().
        self.setup_run(steps)
        if self.run_pre or not self.setup_done:
            self.init_simulation()
        for _ in range(steps):
            self.step()
