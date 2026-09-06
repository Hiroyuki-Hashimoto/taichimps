"""
EXTRA_TAICHI package: High-performance geotechnical & wave propagation extensions.

Provides:
- FixWave: Wave excitation generator (P, S, Rayleigh, Love waves)
- WinSponge / FixSponge: Windowed non-reflecting absorbing boundary
- FixProbe: Sensor array for particle velocity & waveform extraction
- FixTriaxial: 6-face independent stress & strain servo controller
"""

from taichimps.EXTRA_TAICHI.probe import FixAveProbe, FixProbe
from taichimps.EXTRA_TAICHI.sponge import FixSponge, WinSponge
from taichimps.EXTRA_TAICHI.triaxial import FixTriaxial, FixTriaxServo
from taichimps.EXTRA_TAICHI.wave import FixWave

__all__ = [
    "FixAveProbe",
    "FixProbe",
    "FixSponge",
    "FixTriaxServo",
    "FixTriaxial",
    "FixWave",
    "WinSponge",
]
