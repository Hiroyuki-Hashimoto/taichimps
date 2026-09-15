"""
Run a configuration through the real LAMMPS binary and read the result back.

This exists so that "does taichimps match LAMMPS?" can be answered by running
both rather than by reading C++ and hoping.  It drives the CPU build sitting
next to this checkout; tests that use it skip when that binary is absent.

Set TAICHIMPS_LMP to point at a different executable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Two builds sit next to this checkout. build-cpu predates the fix deform/
# pressure h_rate corrections this project reported upstream; build-cpu-hratefix
# is built from the current source tree, which is what taichimps was ported
# from, so that is the one to compare against.
DEFAULT_LMP = Path("/mnt/s_work/hsmt/lammps/build-cpu-hratefix/lmp")
FALLBACK_LMP = Path("/mnt/s_work/hsmt/lammps/build-cpu/lmp")

DUMP_FIELDS = [
    "id", "x", "y", "z",
    "vx", "vy", "vz",
    "omegax", "omegay", "omegaz",
    "fx", "fy", "fz",
    "tqx", "tqy", "tqz",
]


def lmp_executable() -> Path | None:
    """The LAMMPS binary to drive, or None when none is available."""
    env = os.environ.get("TAICHIMPS_LMP")
    if env:
        p = Path(env)
        return p if p.is_file() and os.access(p, os.X_OK) else None
    for candidate in (DEFAULT_LMP, FALLBACK_LMP):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("lmp") or shutil.which("lmp_serial")
    return Path(found) if found else None


def write_data_file(
    path: Path,
    x: np.ndarray,
    radius: np.ndarray,
    density: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    boxlo: tuple[float, float, float],
    boxhi: tuple[float, float, float],
    tilt: tuple[float, float, float] | None = None,
) -> None:
    """Write an `atom_style sphere` data file (diameter + density, as LAMMPS wants)."""
    n = len(x)
    lines = [
        "taichimps parity reference",
        "",
        f"{n} atoms",
        "1 atom types",
        "",
        f"{boxlo[0]:.17g} {boxhi[0]:.17g} xlo xhi",
        f"{boxlo[1]:.17g} {boxhi[1]:.17g} ylo yhi",
        f"{boxlo[2]:.17g} {boxhi[2]:.17g} zlo zhi",
    ]
    if tilt is not None:
        # A triclinic data file carries the tilt factors right after zlo zhi.
        lines.append(f"{tilt[0]:.17g} {tilt[1]:.17g} {tilt[2]:.17g} xy xz yz")
    lines += [
        "",
        "Atoms # sphere",
        "",
    ]
    for i in range(n):
        lines.append(
            f"{i + 1} 1 {2.0 * radius[i]:.17g} {density[i]:.17g} "
            f"{x[i, 0]:.17g} {x[i, 1]:.17g} {x[i, 2]:.17g}"
        )
    lines += ["", "Velocities", ""]
    for i in range(n):
        lines.append(
            f"{i + 1} {v[i, 0]:.17g} {v[i, 1]:.17g} {v[i, 2]:.17g} "
            f"{omega[i, 0]:.17g} {omega[i, 1]:.17g} {omega[i, 2]:.17g}"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def build_input(
    pair_style: str,
    pair_coeff: str | None,
    steps: int,
    dt: float,
    skin: float,
    boundary: str = "p p p",
    integrate: bool = True,
    extra_fixes: str = "",
    pre_pair: str = "",
    box_tilt_large: bool = False,
    thermo_extra: str = "",
    pre_thermo: str = "",
) -> str:
    """
    A minimal granular input.

    `neigh_modify delay 0 every 1 check no` forces a rebuild on every step, so
    that contact history carry-over is exercised rather than bypassed.
    `extra_fixes` is dropped in verbatim, for things like fix deform/pressure.
    """
    fix_line = "fix 1 all nve/sphere" if integrate else ""
    if extra_fixes:
        fix_line = f"{fix_line}\n{extra_fixes}"
    # Even the legacy gran/* styles, whose coefficients are global, still
    # require an explicit pair_coeff line.
    coeff_line = f"pair_coeff {pair_coeff or '* *'}"
    # `comm_modify vel yes` is mandatory for the gran/* styles: they need ghost
    # velocities to evaluate the damping terms.
    return f"""units si
atom_style sphere
boundary {boundary}
newton off
dimension 3
comm_modify mode single vel yes
{"box tilt large" if box_tilt_large else ""}
read_data data.in
{pre_pair}
pair_style {pair_style}
{coeff_line}

neighbor {skin:.17g} bin
neigh_modify delay 0 every 1 check no

timestep {dt:.17g}
{fix_line}

compute vir all pressure NULL pair

dump 1 all custom 1 dump.out {' '.join(DUMP_FIELDS)}
dump_modify 1 sort id format float %.17e

{pre_thermo}
thermo 1
thermo_style custom step lx ly lz c_vir[1] c_vir[2] c_vir[3] {thermo_extra}
# thermo_style resets the formats, so this has to come after it.
thermo_modify format float %.17g

run {steps}
"""


def parse_dump(path: Path) -> list[dict[str, np.ndarray]]:
    """Read every frame of a custom dump into {field: array} dicts, sorted by id."""
    frames: list[dict[str, np.ndarray]] = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        natoms = int(lines[i + 3])
        header = lines[i + 8].split()[2:]
        rows = np.array(
            [[float(tok) for tok in lines[i + 9 + k].split()] for k in range(natoms)]
        )
        frames.append({name: rows[:, c] for c, name in enumerate(header)})
        i += 9 + natoms
    return frames


def run_lammps(
    workdir: Path,
    *,
    x: np.ndarray,
    radius: np.ndarray,
    density: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    boxlo: tuple[float, float, float],
    boxhi: tuple[float, float, float],
    pair_style: str,
    pair_coeff: str | None = None,
    steps: int = 0,
    dt: float = 1e-5,
    skin: float = 0.0,
    boundary: str = "p p p",
    integrate: bool = True,
    extra_fixes: str = "",
    pre_pair: str = "",
    tilt: tuple[float, float, float] | None = None,
    box_tilt_large: bool = False,
    thermo_extra: str = "",
    pre_thermo: str = "",
) -> list[dict[str, np.ndarray]]:
    """Run LAMMPS in `workdir` and return the parsed dump frames."""
    exe = lmp_executable()
    if exe is None:
        raise RuntimeError("No LAMMPS executable available")

    workdir.mkdir(parents=True, exist_ok=True)
    write_data_file(
        workdir / "data.in", x, radius, density, v, omega, boxlo, boxhi, tilt
    )
    (workdir / "in.parity").write_text(
        build_input(
            pair_style, pair_coeff, steps, dt, skin, boundary, integrate,
            extra_fixes, pre_pair, box_tilt_large, thermo_extra, pre_thermo,
        )
    )

    proc = subprocess.run(
        [str(exe), "-in", "in.parity", "-log", "log.lammps"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        log = (workdir / "log.lammps")
        tail = log.read_text()[-4000:] if log.exists() else ""
        raise RuntimeError(
            f"LAMMPS failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}\nlog:\n{tail}"
        )
    return parse_dump(workdir / "dump.out")


def parse_box(path: Path) -> list[np.ndarray]:
    """
    Per-frame box bounds as a (3, 2) array of [lo, hi] per dimension.

    For a triclinic dump LAMMPS reports the *bounding* box plus a tilt factor on
    each line; use parse_tilt() for the tilt and box_from_bound() to recover the
    true bounds.
    """
    boxes: list[np.ndarray] = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("ITEM: BOX BOUNDS"):
            boxes.append(
                np.array([[float(t) for t in lines[i + 1 + d].split()[:2]] for d in range(3)])
            )
    return boxes


def parse_tilt(path: Path) -> list[np.ndarray]:
    """Per-frame [xy, xz, yz], or zeros when the dump is orthogonal."""
    tilts: list[np.ndarray] = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("ITEM: BOX BOUNDS"):
            continue
        if "xy xz yz" in line:
            tilts.append(
                np.array([float(lines[i + 1 + d].split()[2]) for d in range(3)])
            )
        else:
            tilts.append(np.zeros(3))
    return tilts


def box_from_bound(bound: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """
    Undo LAMMPS's triclinic dump convention to get the true [lo, hi] per dim.

    Domain::set_global_box():
      xlo_bound = xlo + min(0, xy, xz, xy+xz), xhi_bound = xhi + max(...)
      ylo_bound = ylo + min(0, yz),            yhi_bound = yhi + max(0, yz)
    """
    xy, xz, yz = (float(v) for v in tilt)
    box = bound.copy()
    box[0, 0] -= min(0.0, xy, xz, xy + xz)
    box[0, 1] -= max(0.0, xy, xz, xy + xz)
    box[1, 0] -= min(0.0, yz)
    box[1, 1] -= max(0.0, yz)
    return box


def force_array(frame: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([frame["fx"], frame["fy"], frame["fz"]], axis=1)


def torque_array(frame: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([frame["tqx"], frame["tqy"], frame["tqz"]], axis=1)


def run_lammps_script(workdir: Path, script: str, name: str = "in.script") -> None:
    """
    Run an arbitrary LAMMPS script in `workdir`.

    `run_lammps` covers the single-shot parity comparisons; this is for cases
    that need several LAMMPS invocations over the same directory, such as a run
    interrupted by write_restart and resumed with read_restart.
    """
    exe = lmp_executable()
    if exe is None:
        raise RuntimeError("No LAMMPS executable available")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / name).write_text(script)
    proc = subprocess.run(
        [str(exe), "-in", name, "-log", f"log.{name}"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        log = workdir / f"log.{name}"
        tail = log.read_text()[-4000:] if log.exists() else ""
        raise RuntimeError(
            f"LAMMPS failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}\nlog:\n{tail}"
        )
