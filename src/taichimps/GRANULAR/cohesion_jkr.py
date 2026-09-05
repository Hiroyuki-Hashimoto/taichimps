"""
JKR Cohesion contact sub-model matching LAMMPS pair_style granular (cohesion_model jkr).
Implements Johnson-Kendall-Roberts adhesive contact mechanics:
  F_pull_off = 1.5 * pi * gamma_surface * r_eff
Reference: LAMMPS src/GRANULAR/gran_sub_mod_cohesion_jkr.cpp
"""

import math
from typing import Any

import taichi as ti


@ti.data_oriented
class CohesionJKR:
    def __init__(
        self,
        surface_energy: float = 0.05, # J/m^2
        float_type: Any = ti.f64,
    ) -> None:
        self.surface_energy = float(surface_energy)
        self.float_type = float_type

    @ti.func
    def compute_cohesive_force(
        self,
        overlap: ti.template(),
        r_eff: ti.template(),
    ):
        """
        Returns attractive cohesive normal force (scalar).
        """
        pi_val = math.pi
        gamma_val = ti.cast(self.surface_energy, self.float_type)
        f_cohesion = 0.0
        if overlap > 0.0:
            # Contact adhesion
            f_cohesion = 1.5 * pi_val * gamma_val * r_eff
        return f_cohesion
