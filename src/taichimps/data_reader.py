"""
LAMMPS Data File Parser.
Supports 'atom_style sphere' data files with Headers, Atoms, and Velocities sections.
Reference: LAMMPS src/read_data.cpp
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LAMMPSData:
    natoms: int
    natom_types: int
    boxlo: list[float]
    boxhi: list[float]
    x: np.ndarray
    radius: np.ndarray
    density: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    tag: np.ndarray
    atom_type: np.ndarray


def read_data(filepath: str | Path) -> LAMMPSData:
    """
    Parse a LAMMPS data file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    natoms = 0
    natom_types = 1
    xlo = ylo = zlo = 0.0
    xhi = yhi = zhi = 1.0

    sections: dict[str, list[str]] = {}
    current_section = "header"
    sections[current_section] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Strip comments and whitespace
            line_str = line.split("#")[0].strip()
            if not line_str:
                continue

            lower_line = line_str.lower()
            if lower_line.startswith("atoms"):
                current_section = "atoms"
                sections[current_section] = []
                continue
            elif lower_line.startswith("velocities"):
                current_section = "velocities"
                sections[current_section] = []
                continue

            sections[current_section].append(line_str)

    # Parse header lines
    for line in sections.get("header", []):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[1] == "atoms":
            natoms = int(tokens[0])
        elif len(tokens) >= 3 and tokens[1] == "atom" and tokens[2] == "types":
            natom_types = int(tokens[0])
        elif len(tokens) >= 4 and tokens[2] == "xlo" and tokens[3] == "xhi":
            xlo, xhi = float(tokens[0]), float(tokens[1])
        elif len(tokens) >= 4 and tokens[2] == "ylo" and tokens[3] == "yhi":
            ylo, yhi = float(tokens[0]), float(tokens[1])
        elif len(tokens) >= 4 and tokens[2] == "zlo" and tokens[3] == "zhi":
            zlo, zhi = float(tokens[0]), float(tokens[1])

    atom_lines = sections.get("atoms", [])
    actual_natoms = len(atom_lines)
    if natoms == 0:
        natoms = actual_natoms

    tags = np.zeros(actual_natoms, dtype=np.int32)
    types = np.zeros(actual_natoms, dtype=np.int32)
    radii = np.zeros(actual_natoms, dtype=np.float64)
    densities = np.zeros(actual_natoms, dtype=np.float64)
    x = np.zeros((actual_natoms, 3), dtype=np.float64)
    v = np.zeros((actual_natoms, 3), dtype=np.float64)
    omega = np.zeros((actual_natoms, 3), dtype=np.float64)

    tag_to_idx: dict[int, int] = {}

    # Format for sphere: id type diameter density x y z
    for idx, line in enumerate(atom_lines):
        toks = line.split()
        t = int(toks[0])
        tags[idx] = t
        types[idx] = int(toks[1])
        # Note: LAMMPS sphere atom format: tag type diameter density x y z
        # radius is diameter / 2.0
        diameter = float(toks[2])
        radii[idx] = 0.5 * diameter
        densities[idx] = float(toks[3])
        x[idx, 0] = float(toks[4])
        x[idx, 1] = float(toks[5])
        x[idx, 2] = float(toks[6])
        tag_to_idx[t] = idx

    vel_lines = sections.get("velocities", [])
    for line in vel_lines:
        toks = line.split()
        t = int(toks[0])
        if t in tag_to_idx:
            idx = tag_to_idx[t]
            v[idx, 0] = float(toks[1])
            v[idx, 1] = float(toks[2])
            v[idx, 2] = float(toks[3])
            if len(toks) >= 7:
                omega[idx, 0] = float(toks[4])
                omega[idx, 1] = float(toks[5])
                omega[idx, 2] = float(toks[6])

    return LAMMPSData(
        natoms=actual_natoms,
        natom_types=natom_types,
        boxlo=[xlo, ylo, zlo],
        boxhi=[xhi, yhi, zhi],
        x=x,
        radius=radii,
        density=densities,
        v=v,
        omega=omega,
        tag=tags,
        atom_type=types,
    )
