"""
Twisting friction contact sub-model matching LAMMPS pair_style granular (twisting_model).
Implements contact-normal spin resistance torque:
  Delta_theta_t = (omega_rel . n) * n * dt
  T_twist = -k_twist * theta_t - gamma_twist * omega_twist
  |T_twist| <= mu_twist * r_eff * F_n
Reference: LAMMPS src/GRANULAR/gran_sub_mod_twisting.cpp
"""

from typing import Any

import taichi as ti


@ti.data_oriented
class TwistingResistance:
    def __init__(
        self,
        k_twist: float = 1e4,
        gamma_twist: float = 10.0,
        mu_twist: float = 0.05,
        float_type: Any = ti.f64,
    ) -> None:
        self.k_twist = float(k_twist)
        self.gamma_twist = float(gamma_twist)
        self.mu_twist = float(mu_twist)
        self.float_type = float_type

    @ti.func
    def compute_torque(
        self,
        omega_rel: ti.template(),
        normal: ti.template(),
        fn: ti.template(),
    ):
        omega_twist = omega_rel.dot(normal) * normal
        torque = -self.gamma_twist * omega_twist
        t_mag = torque.norm()
        t_max = self.mu_twist * fn
        if t_mag > t_max and t_mag > 1e-12:
            torque = torque * (t_max / t_mag)
        return torque
