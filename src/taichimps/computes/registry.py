"""
Named computes, the equivalent of LAMMPS's `compute` command.

Reference: LAMMPS src/compute_pressure.cpp, compute_stress_atom.cpp,
           compute_temp.cpp, compute_ke.cpp, compute_reduce.cpp and
           src/GRANULAR/compute_contact_atom.cpp, compute_fabric.cpp
License: GPL v2 compatible / MIT reimplementation

The input parser used to ignore `compute` entirely and instead fill in a handful
of hard-coded names (`c_1`, `c_3`, `c_5`, `c_6`) with fixed meanings taken from
one particular reference script.  Any script that numbered its computes
differently silently read zeros, and the arguments -- notably the `NULL` that
drops the kinetic term and the `pair` that restricts the virial to pair
interactions -- had no effect at all.

Each compute exposes whichever of `scalar()`, `vector()` and `per_atom()` makes
sense for it, mirroring LAMMPS's scalar_flag / vector_flag / peratom_flag.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, Self

import numpy as np

from taichimps.atom import AtomSystem
from taichimps.computes.contact_atom import ComputeContactAtom
from taichimps.computes.fabric import ComputeFabric
from taichimps.computes.stress_atom import ComputeStressAtom
from taichimps.computes.thermo import Computes
from taichimps.domain import Domain


class ComputeContext(Protocol):
    """What a compute needs from the simulation to evaluate itself."""

    atom: AtomSystem
    domain: Domain
    neighbor: Any


class ComputeVector(float):
    """
    A compute's value as an input script sees it.

    LAMMPS lets the same compute be referenced both ways: `c_1` is the scalar
    and `c_1[1]` the first component of the vector.  Subclassing float gives
    both from one object, and `__getitem__` is 1-based to match the script
    convention.  A compute with no scalar carries NaN, so misusing it shows up
    rather than silently reading zero.
    """

    _values: np.ndarray

    def __new__(cls, scalar: float, values: np.ndarray | None = None) -> Self:
        obj = super().__new__(cls, scalar)
        obj._values = np.asarray([] if values is None else values, dtype=np.float64)
        return obj

    def __getitem__(self, index: int) -> float:
        if len(self._values) == 0:
            raise IndexError("this compute has no vector to index")
        if index < 1 or index > len(self._values):
            raise IndexError(
                f"compute vector index {index} out of range 1..{len(self._values)}"
            )
        return float(self._values[index - 1])

    def __len__(self) -> int:
        return len(self._values)

    def to_numpy(self) -> np.ndarray:
        return self._values


class BaseCompute:
    """Common shape of a named compute."""

    def __init__(self, compute_id: str, style: str) -> None:
        self.id = compute_id
        self.style = style

    def scalar(self, ctx: ComputeContext) -> float:
        raise TypeError(f"compute {self.id} ({self.style}) has no scalar value")

    def vector(self, ctx: ComputeContext) -> ComputeVector:
        raise TypeError(f"compute {self.id} ({self.style}) has no vector value")

    def per_atom(self, ctx: ComputeContext) -> np.ndarray:
        raise TypeError(f"compute {self.id} ({self.style}) has no per-atom values")


class PressureCompute(BaseCompute):
    """
    `compute ID group pressure <temp-ID|NULL> [keyword ...]`

    A NULL temperature compute drops the kinetic term, and listing `pair` (or
    any subset of contributions) restricts the virial -- which here only ever
    has pair contributions anyway.
    """

    def __init__(self, compute_id: str, temp_id: str, keywords: list[str]) -> None:
        super().__init__(compute_id, "pressure")
        self.kinetic = temp_id.upper() != "NULL"
        self.keywords = keywords
        self._computes = Computes()

    def _tensor(self, ctx: ComputeContext) -> np.ndarray:
        return self._computes.compute_pressure_tensor(
            ctx.atom, ctx.domain, kinetic=self.kinetic
        )

    def scalar(self, ctx: ComputeContext) -> float:
        return float(np.mean(self._tensor(ctx)[:3]))

    def vector(self, ctx: ComputeContext) -> ComputeVector:
        tensor = self._tensor(ctx)
        return ComputeVector(float(np.mean(tensor[:3])), tensor)


class TempCompute(BaseCompute):
    """`compute ID group temp` -- the kinetic energy tensor and the temperature."""

    def __init__(self, compute_id: str) -> None:
        super().__init__(compute_id, "temp")
        self._computes = Computes()

    def scalar(self, ctx: ComputeContext) -> float:
        return self._computes.temperature(ctx.atom)

    def vector(self, ctx: ComputeContext) -> ComputeVector:
        return ComputeVector(
            self._computes.temperature(ctx.atom), self._computes.kinetic_tensor(ctx.atom)
        )


class KECompute(BaseCompute):
    """`compute ID group ke` -- translational kinetic energy."""

    def __init__(self, compute_id: str) -> None:
        super().__init__(compute_id, "ke")
        self._computes = Computes()

    def scalar(self, ctx: ComputeContext) -> float:
        return self._computes.ke_trans(ctx.atom)


class ERotateSphereCompute(BaseCompute):
    """`compute ID group erotate/sphere` -- rotational kinetic energy."""

    def __init__(self, compute_id: str) -> None:
        super().__init__(compute_id, "erotate/sphere")
        self._computes = Computes()

    def scalar(self, ctx: ComputeContext) -> float:
        return self._computes.ke_rot(ctx.atom)


class StressAtomCompute(BaseCompute):
    """`compute ID group stress/atom <temp-ID|NULL> [keyword ...]`"""

    def __init__(self, compute_id: str, temp_id: str, keywords: list[str]) -> None:
        super().__init__(compute_id, "stress/atom")
        self.kinetic = temp_id.upper() != "NULL"
        self.keywords = keywords
        self._impl: ComputeStressAtom | None = None

    def per_atom(self, ctx: ComputeContext) -> np.ndarray:
        if self._impl is None:
            self._impl = ComputeStressAtom(max_atoms=ctx.atom.max_atoms)
        return self._impl.compute(ctx.atom, ctx.domain, kinetic=self.kinetic)


class ContactAtomCompute(BaseCompute):
    """`compute ID group contact/atom` -- number of touching neighbours."""

    def __init__(self, compute_id: str) -> None:
        super().__init__(compute_id, "contact/atom")
        self._impl = ComputeContactAtom()

    def per_atom(self, ctx: ComputeContext) -> np.ndarray:
        return self._impl.compute(ctx.atom, ctx.domain, ctx.neighbor).astype(np.float64)


class FabricCompute(BaseCompute):
    """`compute ID group fabric ...` -- the contact fabric tensor."""

    def __init__(self, compute_id: str) -> None:
        super().__init__(compute_id, "fabric")
        self._impl = ComputeFabric()

    def vector(self, ctx: ComputeContext) -> ComputeVector:
        phi, _ = self._impl.compute(ctx.atom, ctx.domain, ctx.neighbor)
        return ComputeVector(float("nan"), phi)


class ReduceCompute(BaseCompute):
    """
    `compute ID group reduce <mode> <input> ...`

    Inputs may be `c_ID`, `c_ID[i]` or `v_name` (an atom-style variable).  The
    reduction runs over the per-atom values of each input; a scalar input is
    taken as-is.
    """

    MODES = ("sum", "ave", "min", "max", "sumsq", "avesq", "sumabs", "aveabs", "maxabs")

    def __init__(
        self,
        compute_id: str,
        mode: str,
        inputs: list[str],
        resolve: Callable[[str, ComputeContext], np.ndarray],
    ) -> None:
        super().__init__(compute_id, "reduce")
        if mode not in self.MODES:
            raise ValueError(
                f"compute reduce mode {mode!r} is not supported; "
                f"implemented: {self.MODES}"
            )
        self.mode = mode
        self.inputs = inputs
        self._resolve = resolve

    def _reduce_one(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        if self.mode == "sum":
            return float(np.sum(values))
        if self.mode == "ave":
            return float(np.mean(values))
        if self.mode == "min":
            return float(np.min(values))
        if self.mode == "max":
            return float(np.max(values))
        if self.mode == "sumsq":
            return float(np.sum(values**2))
        if self.mode == "avesq":
            return float(np.mean(values**2))
        if self.mode == "sumabs":
            return float(np.sum(np.abs(values)))
        if self.mode == "aveabs":
            return float(np.mean(np.abs(values)))
        return float(np.max(np.abs(values)))

    def scalar(self, ctx: ComputeContext) -> float:
        return self._reduce_one(self._resolve(self.inputs[0], ctx))

    def vector(self, ctx: ComputeContext) -> ComputeVector:
        values = np.array(
            [self._reduce_one(self._resolve(i, ctx)) for i in self.inputs]
        )
        return ComputeVector(float(values[0]), values)


def build_compute(
    compute_id: str,
    args: list[str],
    resolve_reduce_input: Callable[[str, ComputeContext], np.ndarray],
) -> BaseCompute:
    """
    Build a compute from its `compute` command arguments (style onwards).

    Unsupported styles raise, rather than quietly producing zeros the way the
    hard-coded table used to.
    """
    style = args[0]
    rest = args[1:]

    if style == "pressure":
        if not rest:
            raise ValueError("compute pressure needs a temperature compute ID or NULL")
        return PressureCompute(compute_id, rest[0], rest[1:])
    if style == "stress/atom":
        if not rest:
            raise ValueError(
                "compute stress/atom needs a temperature compute ID or NULL"
            )
        return StressAtomCompute(compute_id, rest[0], rest[1:])
    if style == "temp":
        return TempCompute(compute_id)
    if style == "ke":
        return KECompute(compute_id)
    if style == "erotate/sphere":
        return ERotateSphereCompute(compute_id)
    if style == "contact/atom":
        return ContactAtomCompute(compute_id)
    if style == "fabric":
        return FabricCompute(compute_id)
    if style == "reduce":
        if len(rest) < 2:
            raise ValueError("compute reduce needs a mode and at least one input")
        return ReduceCompute(compute_id, rest[0], rest[1:], resolve_reduce_input)

    raise ValueError(
        f"Unsupported compute style {style!r}; implemented: pressure, stress/atom, "
        "temp, ke, erotate/sphere, contact/atom, fabric, reduce"
    )
