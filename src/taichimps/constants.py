"""
LAMMPS Mathematical Constants and Units.
Reference: LAMMPS src/math_const.h
License: GPL v2 compatible / MIT reimplementation
"""

import math

# Mathematical constants
MY_PI: float = math.pi
MY_2PI: float = 2.0 * math.pi
MY_PI2: float = 0.5 * math.pi
MY_PI4: float = 0.25 * math.pi
MY_4PI_3: float = 4.0 * math.pi / 3.0

# Unit conversion factors (default 'si')
# LAMMPS si units:
# mass = kilograms
# distance = meters
# time = seconds
# energy = Joules
# velocity = meters/second
# force = Newtons
# torque = Newton-meters
# temperature = Kelvin
# pressure = Pascals
# dynamic viscosity = Pascal*second
