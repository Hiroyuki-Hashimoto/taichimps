"""
LAMMPS atom_style sphere data structure implementation in Taichi.
Reference: LAMMPS src/atom_vec_sphere.cpp
License: LAMMPS GPL v2 / taichimps MIT reimplementation
"""

from typing import Any

import numpy as np
import taichi as ti


@ti.data_oriented
class AtomSystem:
    """
    Manages per-atom fields for granular (atom_style sphere) simulations.
    Fields correspond to LAMMPS atom_vec_sphere:
      - x: position (N, 3)
      - v: velocity (N, 3)
      - f: force (N, 3)
      - omega: angular velocity (N, 3)
      - torque: torque (N, 3)
      - radius: particle radius (N,)
      - rmass: particle mass (N,)
      - atom_type: particle type id (N,) (1-based, like LAMMPS)
      - tag: unique particle ID (N,)
      - mask: group bitmask (N,)
    """

    def __init__(self, max_atoms: int, float_type: Any = ti.f64) -> None:
        self.max_atoms: int = max_atoms
        self.nlocal: int = 0
        self.float_type: Any = float_type

        # Taichi fields (SoA layout for memory bandwidth efficiency)
        self.x = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.v = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.f = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.omega = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.torque = ti.Vector.field(3, dtype=float_type, shape=max_atoms)
        self.radius = ti.field(dtype=float_type, shape=max_atoms)
        self.rmass = ti.field(dtype=float_type, shape=max_atoms)
        self.atom_type = ti.field(dtype=ti.i32, shape=max_atoms)
        self.tag = ti.field(dtype=ti.i32, shape=max_atoms)
        self.mask = ti.field(dtype=ti.i32, shape=max_atoms)

    def add_particles(
        self,
        x: Any,
        radius: Any,
        density: float | Any,
        v: Any | None = None,
        omega: Any | None = None,
        tag: Any | None = None,
        atom_type: int | Any = 1,
    ) -> None:
        """
        Add particles to the system.
        rmass is calculated as (4/3) * pi * r^3 * density.
        """
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, 3)
        n_new = len(x_np)

        if self.nlocal + n_new > self.max_atoms:
            raise ValueError(
                f"Exceeded max_atoms capacity: {self.nlocal + n_new} > {self.max_atoms}"
            )

        start = self.nlocal
        end = start + n_new

        if v is None:
            v_np = np.zeros_like(x_np)
        else:
            v_np = np.asarray(v, dtype=np.float64)
            if v_np.ndim == 1:
                v_np = v_np.reshape(1, 3)

        if omega is None:
            omega_np = np.zeros_like(x_np)
        else:
            omega_np = np.asarray(omega, dtype=np.float64)
            if omega_np.ndim == 1:
                omega_np = omega_np.reshape(1, 3)

        rad_np = np.asarray(radius, dtype=np.float64)
        if rad_np.ndim == 0:
            rad_np = np.full(n_new, rad_np)

        if np.isscalar(density):
            dens_np = np.full(n_new, density, dtype=np.float64)
        else:
            dens_np = np.asarray(density, dtype=np.float64)

        if np.isscalar(atom_type):
            type_np = np.full(n_new, atom_type, dtype=np.int32)
        else:
            type_np = np.asarray(atom_type, dtype=np.int32)

        vol = (4.0 / 3.0) * np.pi * (rad_np**3)
        mass_np = vol * dens_np

        curr_x = self.x.to_numpy()
        curr_v = self.v.to_numpy()
        curr_omega = self.omega.to_numpy()
        curr_rad = self.radius.to_numpy()
        curr_rmass = self.rmass.to_numpy()
        curr_type = self.atom_type.to_numpy()
        curr_tag = self.tag.to_numpy()
        curr_mask = self.mask.to_numpy()

        curr_x[start:end] = x_np
        curr_v[start:end] = v_np
        curr_omega[start:end] = omega_np
        curr_rad[start:end] = rad_np
        curr_rmass[start:end] = mass_np
        curr_type[start:end] = type_np

        if tag is not None:
            curr_tag[start:end] = np.asarray(tag, dtype=np.int32)
        else:
            curr_tag[start:end] = np.arange(start + 1, end + 1, dtype=np.int32)

        curr_mask[start:end] = 1

        self.x.from_numpy(curr_x)
        self.v.from_numpy(curr_v)
        self.omega.from_numpy(curr_omega)
        self.radius.from_numpy(curr_rad)
        self.rmass.from_numpy(curr_rmass)
        self.atom_type.from_numpy(curr_type)
        self.tag.from_numpy(curr_tag)
        self.mask.from_numpy(curr_mask)

        self.nlocal = end

    def filter_particles(self, keep_mask: np.ndarray) -> int:
        """Keep only particles where keep_mask is True, compacting the arrays.

        Args:
            keep_mask: 1D boolean numpy array of length nlocal.

        Returns:
            The new particle count nlocal.
        """
        keep_indices = np.where(keep_mask[: self.nlocal])[0]
        n_kept = len(keep_indices)
        if n_kept == self.nlocal:
            return self.nlocal

        curr_x = self.x.to_numpy()
        curr_v = self.v.to_numpy()
        curr_omega = self.omega.to_numpy()
        curr_f = self.f.to_numpy()
        curr_torque = self.torque.to_numpy()
        curr_rad = self.radius.to_numpy()
        curr_rmass = self.rmass.to_numpy()
        curr_type = self.atom_type.to_numpy()
        curr_tag = self.tag.to_numpy()
        curr_mask = self.mask.to_numpy()

        # Compact kept particles to start
        curr_x[:n_kept] = curr_x[keep_indices]
        curr_v[:n_kept] = curr_v[keep_indices]
        curr_omega[:n_kept] = curr_omega[keep_indices]
        curr_f[:n_kept] = 0.0
        curr_torque[:n_kept] = 0.0
        curr_rad[:n_kept] = curr_rad[keep_indices]
        curr_rmass[:n_kept] = curr_rmass[keep_indices]
        curr_type[:n_kept] = curr_type[keep_indices]
        curr_tag[:n_kept] = curr_tag[keep_indices]
        curr_mask[:n_kept] = 1

        # Clear tail
        curr_mask[n_kept:] = 0

        self.x.from_numpy(curr_x)
        self.v.from_numpy(curr_v)
        self.omega.from_numpy(curr_omega)
        self.f.from_numpy(curr_f)
        self.torque.from_numpy(curr_torque)
        self.radius.from_numpy(curr_rad)
        self.rmass.from_numpy(curr_rmass)
        self.atom_type.from_numpy(curr_type)
        self.tag.from_numpy(curr_tag)
        self.mask.from_numpy(curr_mask)

        self.nlocal = n_kept
        return self.nlocal

    @ti.kernel
    def clear_forces(self):
        """Zero out forces and torques at start of timestep."""
        for i in range(self.nlocal):
            self.f[i] = ti.Vector([0.0, 0.0, 0.0])
            self.torque[i] = ti.Vector([0.0, 0.0, 0.0])
