"""
Abstract Base Class for Granular Pair Styles.
Reference: LAMMPS src/pair.h, src/GRANULAR/pair_gran_hooke_history.h
"""

from abc import ABC, abstractmethod
from typing import Any

import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.contact_history import ContactHistory
from taichimps.domain import Domain
from taichimps.neighbor import NeighborList


class GranularPair(ABC):
    """Base class for all granular pair potentials."""

    def __init__(self, domain: Domain, float_type: Any = ti.f64) -> None:
        self.domain = domain
        self.float_type = float_type

    @abstractmethod
    def compute(
        self,
        atom: AtomSystem,
        nlist: NeighborList,
        history: ContactHistory,
        dt: float,
    ) -> None:
        """Compute contact forces and torques and update history."""
