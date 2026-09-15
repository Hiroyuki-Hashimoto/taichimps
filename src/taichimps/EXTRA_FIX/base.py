"""
Abstract Base Class for Fixes.
Reference: LAMMPS src/fix.h
"""

from abc import ABC
from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.domain import Domain


class Fix(ABC):
    """Base class for all Fix implementations."""

    def __init__(self, domain: Domain | None = None, float_type: Any = ti.f64) -> None:
        self.domain = domain
        self.float_type = float_type
        # LAMMPS clears this while update->setupflag is set, so the setup force
        # evaluation does not advance any contact history a fix may hold.
        self.history_update = True
        # Set by a fix that changed the box in a way the neighbor list cannot
        # tolerate until atoms are remapped -- a triclinic box flip. Simulation
        # clears it once it has rebuilt. LAMMPS calls the same thing
        # `next_reneighbor`.
        self.force_reneighbor = False

    def setup(self, nsteps_total: int = 0, dt: float = 0.0) -> None:
        """
        Called once at the start of a run, the analogue of LAMMPS Fix::init().

        Fixes that reference the state of the system at the start of a run
        (fix deform latching the box, for instance) latch it here, and those
        that interpolate towards a target at the end of the run get the run
        length and the timestep they need.
        """

    def initial_integrate(self, atom: AtomSystem, dt: float) -> None:
        """Called at beginning of timestep (e.g. Velocity Verlet half-step)."""

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        """Called after pair forces are evaluated (e.g. gravity, walls)."""

    def final_integrate(self, atom: AtomSystem, dt: float) -> None:
        """Called at end of timestep (e.g. Velocity Verlet 2nd half-step)."""

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        """Called at the end of each simulation timestep."""

