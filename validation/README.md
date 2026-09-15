# Validation against LAMMPS

Long-running comparisons that are too slow to live in `tests/`, kept here so
they can be re-run whenever the core changes.

## FCC triaxial (`fcc_triaxial.py`)

A 1400-sphere Thornton FCC packing, isotropically compressed to 100 kPa and
then sheared in a strain-controlled triaxial test at constant lateral pressure,
for six friction coefficients.  It is the case this project exists to run, and
it exercises nearly everything at once: `pair granular` with
`hertz/material` + `mindlin` + `coeff_restitution`, `fix deform/pressure` with
two pressure-servoed axes and one strain-rate axis under `remap v`,
`read_restart`, the contact-history carry-over, and the whole compute and
`fix print` output path.

**taichimps runs the LAMMPS input script itself.** There is no separate
taichimps input: `in.triax` is handed to `taichimps.input.LAMMPSInputParser`
unchanged, so the two codes cannot silently drift apart in their settings, and
the parser is under test too.

### What it checks

* stress (`pxx`, `pyy`, `pzz`), box lengths, void ratio, mean coordination
  number and kinetic energy against the LAMMPS run, step by step
* the peak stress ratio against Thornton's analytical value for an FCC lattice,
  `sigma_1 / sigma_3 = 2 (1 + mu) / (1 - mu)`

### Running it

The reference LAMMPS output and the starting restart file live outside this
repository, under `lammps-work/20260912-FCCtest/`.  Point `--reference` at that
directory.

```
# one friction coefficient, the first 30000 steps -- a few minutes, and enough
# to catch anything that has broken
uv run python validation/fcc_triaxial.py --mu 0.3 --steps 30000

# the full shear, to 0.1% axial strain (5e6 steps, roughly an hour per case)
uv run python validation/fcc_triaxial.py --mu 0.3

# every friction coefficient
uv run python validation/fcc_triaxial.py --all
```

Results land in `validation/results/<case>/`, next to a `comparison.txt`
holding the step-by-step differences and the peak stress ratio.
