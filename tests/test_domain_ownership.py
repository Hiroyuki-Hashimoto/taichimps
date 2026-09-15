"""
Who owns the current box, the host or the device.

Once a kernel updates the box, the host's numpy mirror is out of date. Reading
it anyway is a silent wrong answer -- the run keeps going and produces slightly
wrong output -- so the mirror refuses to be read until it has been pulled back.

The refusal exists to protect the per-step path. Pulling costs a device
synchronisation (~80 us measured on CUDA); one such pull on a path that runs
every step would undo the reason for putting the box on the device at all. The
neighbour list's displacement check was exactly that kind of innocent-looking
per-step read and cost 0.48 ms per step until it was found.
"""

import numpy as np
import pytest
import taichi as ti

from taichimps.domain import Domain

GUARDED = ("boxlo", "boxhi", "prd", "tilt", "h_rate", "volume", "triclinic",
           "xy", "xz", "yz")


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def _domain():
    return Domain(boxlo=[0.0, 0.0, 0.0], boxhi=[2.0, 3.0, 4.0],
                  boundary=("p", "p", "f"), tilt=[0.1, 0.2, 0.3])


def test_the_mirror_reads_normally_while_the_host_owns_the_box():
    d = _domain()
    np.testing.assert_allclose(d.prd, [2.0, 3.0, 4.0])
    assert d.volume == pytest.approx(24.0)
    assert d.triclinic


@pytest.mark.parametrize("name", GUARDED)
def test_reading_a_stale_mirror_is_refused(name):
    d = _domain()
    d.mark_device_authoritative()
    with pytest.raises(RuntimeError, match="pull"):
        getattr(d, name)


def test_the_boundary_conditions_stay_readable():
    """Kernels never change them, so the host is always entitled to answer."""
    d = _domain()
    d.mark_device_authoritative()
    np.testing.assert_array_equal(d.periodicity, [1, 1, 0])


def test_pull_restores_the_mirror_from_the_device():
    d = _domain()
    d.set_box_and_h_rate([1.0, 1.0, 1.0], [5.0, 7.0, 9.0], [0.4, 0.5, 0.6],
                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vremap=True)
    d.mark_device_authoritative()
    d.pull()
    np.testing.assert_allclose(d.boxlo, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(d.boxhi, [5.0, 7.0, 9.0])
    np.testing.assert_allclose(d.prd, [4.0, 6.0, 8.0])
    np.testing.assert_allclose(d.tilt, [0.4, 0.5, 0.6])
    np.testing.assert_allclose(d.h_rate, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert d.volume == pytest.approx(192.0)


def test_a_second_pull_costs_nothing_and_changes_nothing():
    d = _domain()
    d.mark_device_authoritative()
    d.pull()
    before = d.prd.copy()
    d.pull()
    np.testing.assert_array_equal(d.prd, before)


def test_the_host_setting_the_box_clears_the_staleness():
    d = _domain()
    d.mark_device_authoritative()
    d.set_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(d.prd, [1.0, 1.0, 1.0])
