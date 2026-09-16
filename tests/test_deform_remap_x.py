"""
`fix deform/pressure` with `remap x`, the LAMMPS default.

Every other test, and the FCC validation case, writes `remap v` explicitly, so
the affine path went unexercised -- and it was broken on CUDA.  The kernel read
six zero-dimensional box fields at kernel scope and then looped over a run-time
bound; Taichi splits that into a serial prologue and a parallel launch passing
values through a temporary, and the pair miscompiled into writes outside the
kernel's own fields.  Nothing failed at the point of damage: the next
synchronisation, in the unrelated neighbour-rebuild check, raised
CUDA_ERROR_MISALIGNED_ADDRESS.

So this is checked twice: on the CPU for the arithmetic, and on CUDA -- where
the fault lived -- for the fault.
"""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

RADIUS, DENSITY, DT = 5.0e-4, 2650.0, 1.0e-7
ERATE = -0.05


def _run(arch, nsteps: int = 30):
    ti.init(arch=arch, default_fp=ti.f64)

    from taichimps.atom import AtomSystem
    from taichimps.domain import Domain
    from taichimps.contact_history import ContactHistory
    from taichimps.EXTRA_FIX.deform_pressure import FixDeformPressure
    from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
    from taichimps.GRANULAR.hertz_history import GranHertzHistory
    from taichimps.neighbor import NeighborList
    from taichimps.simulation import Simulation

    # A loose cubic block: what matters is that the box moves and the atoms
    # have to move with it, not that anything is in contact.
    n_side = 6
    lo, hi = 0.0, n_side * 2.5 * RADIUS
    g = (np.arange(n_side) + 0.5) * (hi - lo) / n_side
    x = np.array([[a, b, c] for a in g for b in g for c in g])
    n = len(x)

    dom = Domain(boxlo=[lo] * 3, boxhi=[hi] * 3, boundary=("p", "p", "p"))
    atom = AtomSystem(max_atoms=n)
    atom.add_particles(x=x, radius=np.full(n, RADIUS), density=np.full(n, DENSITY))
    nlist = NeighborList(domain=dom, max_atoms=n, max_neighbors_per_atom=32,
                         skin=0.4 * RADIUS)
    hist = ContactHistory(max_atoms=n, max_neighbors=32)
    pair = GranHertzHistory(domain=dom, kn=1e8, gamman=0.0, kt=1e8, gammat=0.0,
                            xmu=0.3, dampflag=0)
    sim = Simulation(domain=dom, atom=atom, neighbor=nlist, history=hist,
                     pair=pair, dt=DT)
    sim.add_fix(FixNVESphere(domain=dom))
    sim.add_fix(FixDeformPressure(
        domain=dom, nevery=1, remap="x",
        axes={"z": {"style": "erate", "rate": ERATE}}))

    sim.run(nsteps)
    ti.sync()
    dom.pull()
    return {
        "x": atom.x.to_numpy()[:n],
        "boxlo": np.asarray(dom.boxlo).copy(),
        "boxhi": np.asarray(dom.boxhi).copy(),
        "x0": x,
        "lo0": np.full(3, lo),
        "hi0": np.full(3, hi),
        "elapsed": nsteps * DT,
    }


def _check_affine(r):
    """
    An affine remap is exactly the statement that every atom keeps its
    fractional position in the box, so that is what is asserted -- along with
    the box having actually moved, so that the check is not vacuous.
    """
    prd0 = r["hi0"] - r["lo0"]
    prd = r["boxhi"] - r["boxlo"]

    assert np.isfinite(r["x"]).all(), "remap produced non-finite coordinates"

    # z is driven at a constant engineering strain rate; x and y are free.
    expected_z = prd0[2] * (1.0 + ERATE * r["elapsed"])
    assert prd[2] == pytest.approx(expected_z, rel=1e-9), (
        f"driven axis is {prd[2]!r}, expected {expected_z!r}")
    assert prd[2] < prd0[2], "the box did not contract"
    assert prd[:2] == pytest.approx(prd0[:2], rel=1e-12), (
        "an axis with no style was deformed")

    lam0 = (r["x0"] - r["lo0"]) / prd0
    lam = (r["x"] - r["boxlo"]) / prd
    # The atoms are free particles under gravity-free NVE with no contacts at
    # this spacing, so the only thing that moved them is the remap itself.
    assert np.abs(lam - lam0).max() < 1e-9, (
        "remap x did not carry the atoms with the box: fractional positions "
        f"moved by {np.abs(lam - lam0).max():.3e}")


def test_remap_x_moves_atoms_with_the_box_on_cpu():
    _check_affine(_run(ti.cpu))


@pytest.mark.skipif(not ti._lib.core.with_cuda(), reason="no CUDA device")
def test_remap_x_survives_on_cuda():
    """
    The regression proper.  Before the fix this did not return a wrong answer,
    it aborted the process, so simply completing is most of the assertion.
    """
    _check_affine(_run(ti.cuda))
