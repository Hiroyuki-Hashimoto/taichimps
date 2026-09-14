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
        shearupdate: bool = True,
    ) -> None:
        """
        Compute contact forces, torques and the pairwise virial, and update history.

        `shearupdate` mirrors the LAMMPS flag of the same name: it is False for
        the setup force evaluation, where the shear history must not advance.
        """
