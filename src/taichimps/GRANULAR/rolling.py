"""
Rolling friction contact sub-model matching LAMMPS pair_style granular (rolling_model).
Implements EPSD (Elastic-Plastic Spring Dashpot) model for rolling resistance torque:
  Delta_theta_r = (omega_i - omega_j) x r_ij * dt
  T_r = -k_roll * theta_r - gamma_roll * omega_rel
  |T_r| <= mu_roll * r_eff * F_n
Reference: LAMMPS src/GRANULAR/gran_sub_mod_rolling.cpp
"""

from typing import Any

import taichi as ti


@ti.data_oriented
class RollingResistance:
    def __init__(
        self,
        k_roll: float = 1e4,
        gamma_roll: float = 10.0,
        mu_roll: float = 0.05,
        float_type: Any = ti.f64,
    ) -> None:
        self.k_roll = float(k_roll)
        self.gamma_roll = float(gamma_roll)
        self.mu_roll = float(mu_roll)
        self.float_type = float_type

    @ti.func
    def compute_torque(
        self,
        omega_rel: ti.template(),
        rolling_hist: ti.template(),
        r_eff: ti.template(),
        fn: ti.template(),
        dt: ti.f64,
    ):
        """
        Returns: (torque_i, new_rolling_hist)
        """
        new_hist = rolling_hist + omega_rel * dt
        tr_mag = self.k_roll * new_hist.norm()
        tr_max = self.mu_roll * r_eff * fn

        torque = -self.k_roll * new_hist - self.gamma_roll * omega_rel
        if tr_mag > tr_max and tr_mag > 1e-12:
            torque = -tr_max * (new_hist / tr_mag)
            new_hist = (tr_max / self.k_roll) * (new_hist / tr_mag)

        return torque, new_hist
