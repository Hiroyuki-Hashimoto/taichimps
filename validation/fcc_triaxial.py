#!/usr/bin/env python
"""
Run the FCC triaxial case in taichimps and compare it with LAMMPS.

See validation/README.md.  The short version: the LAMMPS input script is handed
to the taichimps parser unchanged, so both codes run the same settings by
construction, and the outputs are compared file for file.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = REPO.parent / "lammps-work" / "20260912-FCCtest"

# The quantities each `fix print` file carries, in column order.
OUTPUTS: dict[str, list[str]] = {
    "mean_stress.txt": ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"],
    "p_qx_qy_qz.txt": ["p", "qx", "qy", "qz"],
    "lx_ly_lz.txt": ["lx", "ly", "lz"],
    "coordnos.txt": ["cn"],
    "voidratio.txt": ["e"],
    "energy_term.txt": ["KEtra", "KErot", "KEall"],
}

# Rotational kinetic energy is compared loosely on purpose.  An FCC lattice
# sheared along a lattice axis develops no net spin, so KErot sits at ~1e-27 J,
# which is numerically zero: the two codes agree on it to every digit that
# means anything, and the relative difference is pure rounding.
LOOSE = {"KErot"}
FLOOR = {"KEtra": 1e-20, "KErot": 1e-24, "KEall": 1e-20,
         "pxy": 1.0, "pxz": 1.0, "pyz": 1.0}


def case_dir(reference: Path, mu: float) -> Path:
    """The LAMMPS run directory for a friction coefficient."""
    tag = f"{mu:.1f}".replace(".", "p")
    candidates = sorted(reference.glob(f"triax_*_mu{tag}_*hratefix"))
    if not candidates:
        candidates = sorted(reference.glob(f"triax_*_mu{tag}_*"))
    if not candidates:
        raise SystemExit(f"no LAMMPS reference run for mu={mu} under {reference}")
    return candidates[-1]


def prepare(work: Path, lammps_case: Path, reference: Path, steps: int | None,
            every: int | None) -> Path:
    """Copy the LAMMPS script and its restart into a clean working directory."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    script = (lammps_case / "in.triax").read_text()
    restart = re.search(r"^read_restart\s+(\S+)", script, re.MULTILINE)
    if not restart:
        raise SystemExit(f"{lammps_case}/in.triax has no read_restart line")
    name = restart.group(1)
    for src in (lammps_case / name, reference / "iso_100kPa" / name):
        if src.is_file():
            shutil.copy(src, work / name)
            break
    else:
        raise SystemExit(f"cannot find the restart file {name}")

    if steps is not None:
        script = re.sub(r"^run\s+\d+.*$", f"run {steps}", script, flags=re.MULTILINE)
    if every is not None:
        script = re.sub(r"(^fix\s+\S+\s+\S+\s+print\s+)\d+", rf"\g<1>{every}",
                        script, flags=re.MULTILINE)
    out = work / "in.triax"
    out.write_text(script)
    return out


def run_taichimps(script: Path, arch: str, fp: str) -> float:
    """Execute the script through the taichimps parser. Returns seconds elapsed."""
    import taichi as ti

    from taichimps.input import LAMMPSInputParser

    cwd = Path.cwd()
    try:
        # fix print writes relative to the script, but dumps and the parser's
        # workdir resolution are happier with the working directory set too.
        import os

        os.chdir(script.parent)
        parser = LAMMPSInputParser(
            script,
            default_fp=ti.f64 if fp == "f64" else ti.f32,
            arch=getattr(ti, arch),
        )
        t0 = time.perf_counter()
        parser.execute()
        ti.sync()
        return time.perf_counter() - t0
    finally:
        import os

        os.chdir(cwd)


def load(path: Path) -> dict[int, list[float]]:
    """A `fix print` output file as {timestep: [values]}."""
    rows: dict[int, list[float]] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            rows[int(float(parts[0]))] = [float(v) for v in parts[1:]]
        except ValueError:
            continue
    return rows


def compare(lammps_case: Path, work: Path, mu: float, tol: float) -> tuple[str, bool]:
    """Compare every output file. Returns (report, ok)."""
    lines: list[str] = []
    ok = True

    for name, columns in OUTPUTS.items():
        ref_file, got_file = lammps_case / name, work / name
        if not ref_file.is_file() or not got_file.is_file():
            lines.append(f"{name}: missing on one side, skipped")
            continue
        ref, got = load(ref_file), load(got_file)
        common = sorted(set(ref) & set(got))
        if not common:
            lines.append(f"{name}: no timesteps in common")
            ok = False
            continue
        for c, column in enumerate(columns):
            errs = []
            for step in common:
                if c >= len(ref[step]) or c >= len(got[step]):
                    continue
                a, b = ref[step][c], got[step][c]
                scale = max(abs(a), FLOOR.get(column, 0.0))
                if scale == 0.0:
                    continue
                errs.append((abs(a - b) / scale, step))
            if not errs:
                continue
            e, step = max(errs)
            limit = 1.0 if column in LOOSE else tol
            flag = "" if e < limit else "   <-- OVER TOLERANCE"
            if e >= limit:
                ok = False
            lines.append(
                f"  {column:6s} max rel {e:9.2e} at step {step:>9d}"
                f"  over {len(common)} samples{flag}"
            )

    # Peak stress ratio against Thornton's analytical FCC value.
    ref, got = load(lammps_case / "mean_stress.txt"), load(work / "mean_stress.txt")
    theory = 2.0 * (1.0 + mu) / (1.0 - mu)

    def ratios(rows: dict[int, list[float]], steps: set[int] | None = None):
        return [
            (r[2] / (0.5 * (r[0] + r[1])), s)
            for s, r in sorted(rows.items())
            if len(r) >= 3 and r[0] + r[1] != 0.0 and (steps is None or s in steps)
        ]

    lines.append("")
    lines.append(f"peak stress ratio (sigma_1/sigma_3), mu = {mu}")
    lines.append(f"  theory 2(1+mu)/(1-mu)              {theory:8.4f}")
    shared = set(ref) & set(got)
    if shared:
        # Over the steps both sides actually cover, so that a deliberately
        # shortened taichimps run is not compared against the full LAMMPS
        # shear curve and reported as having peaked early.
        lo, hi = min(shared), max(shared)
        lines.append(f"  over the shared range {lo}-{hi}:")
        for label, rows in (("LAMMPS", ref), ("taichimps", got)):
            peak, at = max(ratios(rows, shared))
            lines.append(f"    {label:22s} {peak:8.4f}   at step {at}")
    full_peak, full_at = max(ratios(ref))
    lines.append(
        f"  LAMMPS over its whole run:        {full_peak:8.4f}   at step {full_at}"
        f"   ({100.0 * full_peak / theory:5.1f}% of theory)"
    )
    if shared and max(shared) < full_at:
        lines.append(
            "  (the taichimps run stops before the peak; re-run without --steps"
            " to compare peaks)"
        )
    else:
        peak, at = max(ratios(got))
        lines.append(
            f"  taichimps over its whole run:     {peak:8.4f}   at step {at}"
            f"   ({100.0 * peak / theory:5.1f}% of theory)"
        )
    return "\n".join(lines), ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mu", type=float, default=0.3,
                    help="friction coefficient of the case to run")
    ap.add_argument("--all", action="store_true",
                    help="run every friction coefficient the reference has")
    ap.add_argument("--steps", type=int, default=None,
                    help="shorten the run (default: whatever in.triax says)")
    ap.add_argument("--every", type=int, default=None,
                    help="override the fix print interval")
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                    help="directory holding the LAMMPS runs")
    ap.add_argument("--results", type=Path, default=REPO / "validation" / "results")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--fp", default="f64", choices=("f64", "f32"))
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative tolerance for the step-by-step comparison")
    args = ap.parse_args()

    mus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] if args.all else [args.mu]
    all_ok = True
    for mu in mus:
        lammps_case = case_dir(args.reference, mu)
        work = args.results / lammps_case.name
        script = prepare(work, lammps_case, args.reference, args.steps, args.every)

        print(f"\n=== mu = {mu}  ({lammps_case.name}) ===", flush=True)
        elapsed = run_taichimps(script, args.arch, args.fp)
        nsteps = int(re.search(r"^run\s+(\d+)", script.read_text(), re.MULTILINE).group(1))
        print(f"  taichimps: {elapsed:.1f} s for {nsteps} steps "
              f"({elapsed / nsteps * 1e3:.4f} ms/step)", flush=True)

        report, ok = compare(lammps_case, work, mu, args.tol)
        print(report, flush=True)
        (work / "comparison.txt").write_text(
            f"mu = {mu}\nLAMMPS reference: {lammps_case}\n"
            f"taichimps: {elapsed:.1f} s for {nsteps} steps\n\n{report}\n"
        )
        all_ok = all_ok and ok

    print("\nAGREES" if all_ok else "\nDISAGREES -- see the lines marked OVER TOLERANCE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
