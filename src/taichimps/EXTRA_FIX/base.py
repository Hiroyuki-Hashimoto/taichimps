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

    def initial_integrate(self, atom: AtomSystem, dt: float) -> None:
        """Called at beginning of timestep (e.g. Velocity Verlet half-step)."""

    def post_force(self, atom: AtomSystem, dt: float) -> None:
        """Called after pair forces are evaluated (e.g. gravity, walls)."""

    def final_integrate(self, atom: AtomSystem, dt: float) -> None:
        """Called at end of timestep (e.g. Velocity Verlet 2nd half-step)."""

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        """Called at the end of each simulation timestep."""

