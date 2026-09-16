"""
`pair_style granular` has to survive the input parser, not just exist as a class.

Before this, input.py only recognised the three legacy gran/* styles and
ignored pair_coeff entirely, so a script written against pair_style granular --
which is what the reference triaxial inputs use -- left pair_style as None and
ran with no contact forces whatsoever.
"""

import math

import numpy as np
import pytest
import taichi as ti

from taichimps.GRANULAR.granular import (
    PairGranular,
    mix_stiffness_e,
    mix_stiffness_g,
)
from taichimps.input import LAMMPSInputParser

EMOD = 71.6e9
POISS = 0.23
COR = 0.95
FRIC = 0.3

# The box comes from the script rather than being injected afterwards:
# LAMMPSInputParser.execute() calls ti.init(), which tears down any Taichi
# fields allocated before it -- including the ones Domain now holds.
SCRIPT = f"""
units           si
boundary        p p p
atom_style      sphere
dimension       3

region          box block 0.0 1.0 0.0 1.0 0.0 1.0
create_box      1 box

variable        Ep equal {EMOD}
variable        Poi equal {POISS}
variable        Cr equal {COR}
variable        fric equal {FRIC}

pair_style      granular
pair_coeff      * * hertz/material ${{Ep}} ${{Cr}} ${{Poi}} \
tangential mindlin NULL 1.0 ${{fric}} damping coeff_restitution
"""


@pytest.fixture(scope="module", autouse=True)
def _init_taichi():
    ti.init(arch=ti.cpu, default_fp=ti.f64)


def test_pair_style_granular_is_parsed(tmp_path):
    script = tmp_path / "in.granular"
    script.write_text(SCRIPT)

    parser = LAMMPSInputParser(script, default_fp=ti.f64, arch=ti.cpu)
    parser.execute()

    pair = parser.pair_style
    assert isinstance(pair, PairGranular), (
        "pair_style granular did not produce a pair style; "
        "the script would have run with no contact forces"
    )

    # Stiffnesses derived the way LAMMPS does in coeffs_to_local().
    expected_kn = (4.0 / 3.0) * mix_stiffness_e(EMOD, EMOD, POISS, POISS)
    expected_kt = 8.0 * mix_stiffness_g(EMOD, EMOD, POISS, POISS)
    np.testing.assert_allclose(pair.kn, expected_kn, rtol=1e-12)
    np.testing.assert_allclose(pair.kt, expected_kt, rtol=1e-12)
    assert pair.mu == pytest.approx(FRIC)

    # GranSubModDampingCoeffRestitution::init(), Hertzian branch.
    logcor = math.log(COR)
    expected_damp = (
        -1.22474487139158894067
        * 1.82574185835055380345
        * logcor
        / math.sqrt(math.pi**2 + logcor**2)
    )
    np.testing.assert_allclose(pair.damp, expected_damp, rtol=1e-12)


def test_unsupported_submodel_is_rejected(tmp_path):
    """An unimplemented sub-model must fail loudly, not fall back to something else."""
    script = tmp_path / "in.jkr"
    script.write_text(
        "units si\nboundary p p p\natom_style sphere\n"
        "region box block 0.0 1.0 0.0 1.0 0.0 1.0\ncreate_box 1 box\n"
        "pair_style granular\n"
        "pair_coeff * * jkr 1e9 0.95 0.23 0.1 tangential mindlin NULL 1.0 0.3\n"
    )
    parser = LAMMPSInputParser(script, default_fp=ti.f64, arch=ti.cpu)
    with pytest.raises(ValueError, match="jkr"):
        parser.execute()
