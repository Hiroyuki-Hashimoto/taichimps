"""
LAMMPS Input Script Parser and Runner for taichimps.
Reference: LAMMPS src/input.cpp, src/variable.cpp
License: GPL v2 compatible / MIT reimplementation
"""

import math
import re
import shlex
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import taichi as ti

from taichimps.atom import AtomSystem
from taichimps.data_reader import read_data
from taichimps.domain import Domain
from taichimps.dump import DumpWriter
from taichimps.EXTRA_FIX.base import Fix
from taichimps.EXTRA_FIX.damping_cundall import FixDampingCundall
from taichimps.EXTRA_FIX.deform_pressure import FixDeformPressure
from taichimps.EXTRA_FIX.drag import FixDrag
from taichimps.EXTRA_FIX.freeze import FixFreeze
from taichimps.EXTRA_FIX.gravity import FixGravity
from taichimps.EXTRA_FIX.nve_sphere import FixNVESphere
from taichimps.EXTRA_FIX.print import FixPrint
from taichimps.EXTRA_FIX.viscous_sphere import FixViscousSphere
from taichimps.EXTRA_FIX.wall_gran import FixWallGran
from taichimps.EXTRA_FIX.wall_gran_region import FixWallGranRegion
from taichimps.EXTRA_TAICHI.probe import FixProbe
from taichimps.EXTRA_TAICHI.sponge import WinSponge
from taichimps.EXTRA_TAICHI.triaxial import FixTriaxial
from taichimps.EXTRA_TAICHI.wave import FixWave
from taichimps.GRANULAR.granular import PairGranular
from taichimps.GRANULAR.hertz_history import GranHertzHistory
from taichimps.GRANULAR.hooke import GranHooke
from taichimps.GRANULAR.hooke_history import GranHookeHistory
from taichimps.neighbor import NeighborList
from taichimps.simulation import Simulation


class LAMMPSInputParser:
    """Parses and executes a LAMMPS input script in taichimps."""

    def __init__(self, script_path: str | Path, default_fp=None, arch=None) -> None:
        self.script_path = Path(script_path)
        self.workdir = self.script_path.parent
        self.variables: dict[str, Any] = {}
        # Expressions of equal-style variables, kept so they can be
        # re-evaluated on demand. LAMMPS re-evaluates an equal-style variable
        # every time it is referenced; storing only the value frozen at
        # definition time makes any time-dependent variable a constant.
        self.variable_exprs: dict[str, str] = {}
        self.commands: list[str] = []
        self.default_fp = default_fp
        self.arch = arch

        # System state
        self.domain: Domain | None = None
        self.atom: AtomSystem | None = None
        self.neighbor: NeighborList | None = None
        self.simulation: Simulation | None = None
        self.dt: float = 1e-4
        self.thermo_freq: int = 100
        self.pair_style: Any = None
        self.fixes: dict[str, Any] = {}
        self.dumps: dict[str, Any] = {}
        self.regions: dict[str, Any] = {}
        self.custom_thermo_keys: list[str] = []
        self.load_script()

    def evaluate_string_template(self, template_str: str) -> str:
        """Expand variable references inside a string template (e.g. for fix print)."""
        self.update_dynamic_variables()

        def sub_var(match: re.Match) -> str:
            var_name = match.group(1)
            val = self.variables.get(var_name, "")
            return str(val)

        res = re.sub(r"\$\{([a-zA-Z0-9_]+)\}", sub_var, template_str)
        res = re.sub(r"\$([a-zA-Z0-9_]+)", sub_var, res)
        return res

    def update_dynamic_variables(self) -> None:
        """Update system state variables like step, lx, ly, lz, vol, c_1, etc."""
        if self.simulation is not None:
            self.variables["step"] = float(self.simulation.timestep)
            self.variables["lstep"] = float(self.simulation.timestep)
            self.variables["dt"] = float(self.simulation.dt)
        if self.domain is not None:
            lx = float(self.domain.prd[0])
            ly = float(self.domain.prd[1])
            lz = float(self.domain.prd[2])
            vol = float(self.domain.volume)
            self.variables["lx"] = lx
            self.variables["ly"] = ly
            self.variables["lz"] = lz
            self.variables["xlength"] = lx
            self.variables["ylength"] = ly
            self.variables["zlength"] = lz
            self.variables["vol"] = vol
        if self.atom is not None and self.domain is not None and self.simulation is not None:
            # Solid volume and void ratio
            rad = self.atom.radius.to_numpy()[: self.atom.nlocal]
            solid_vol = float(np.sum(4.0 / 3.0 * np.pi * (rad**3)))
            self.variables["c_4"] = solid_vol
            vol = float(self.domain.volume)
            self.variables["vol"] = vol
            self.variables["voidratio"] = (vol - solid_vol) / solid_vol if solid_vol > 0 else 0.0
            # Kinetic energies
            ke_t = self.simulation.computes.ke_trans(self.atom)
            ke_r = self.simulation.computes.ke_rot(self.atom)
            self.variables["ke"] = ke_t + ke_r
            self.variables["ke_rot"] = ke_r
            self.variables["ke_trans"] = ke_t

            # Pressures
            p_tensor = self.simulation.computes.compute_pressure_tensor(self.atom, self.domain)
            self.variables["pxx"] = float(p_tensor[0])
            self.variables["pyy"] = float(p_tensor[1])
            self.variables["pzz"] = float(p_tensor[2])
            self.variables["pxy"] = float(p_tensor[3])
            self.variables["pxz"] = float(p_tensor[4])
            self.variables["pyz"] = float(p_tensor[5])
            p_mean = (p_tensor[0] + p_tensor[1] + p_tensor[2]) / 3.0
            self.variables["p"] = float(p_mean)
            # The *_p variables stand for the pair-only pressure, i.e. what
            # `compute <id> all pressure NULL pair` reports: passing NULL for
            # the temperature compute drops the kinetic term entirely.
            p_pair = self.simulation.computes.compute_pressure_tensor(
                self.atom, self.domain, kinetic=False
            )
            self.variables["pxx_p"] = float(p_pair[0])
            self.variables["pyy_p"] = float(p_pair[1])
            self.variables["pzz_p"] = float(p_pair[2])

            # Energy
            ke_t = self.simulation.computes.ke_trans(self.atom)
            ke_r = self.simulation.computes.ke_rot(self.atom)
            ke_tot = ke_t + ke_r
            self.variables["c_5"] = ke_t
            self.variables["KEtra"] = ke_t
            self.variables["c_6"] = ke_r
            self.variables["KErot"] = ke_r
            self.variables["KEall"] = ke_tot

            # Coordination number
            nlist = self.neighbor or self.simulation.neighbor
            coord_nums = self.simulation.computes.compute_coordination_number(self.atom, nlist)
            cn_mean = float(np.mean(coord_nums[:self.atom.nlocal])) if self.atom.nlocal > 0 else 0.0
            self.variables["c_3"] = cn_mean
            self.variables["cn"] = cn_mean

            # Also evaluate any equal variables that depend on dynamic variables
            # e.g. variable e equal (${vol}-${Vs})/${Vs}
            if "Vs" in self.variables:
                vs = float(self.variables["Vs"])
                if vs > 0.0:
                    self.variables["e"] = (self.domain.volume - vs) / vs

    # Coefficient counts per sub-model, from the LAMMPS num_coeffs fields in
    # gran_sub_mod_{normal,damping,tangential}.cpp.
    _GRANULAR_NCOEFF: ClassVar[dict[str, dict[str, int]]] = {
        "normal": {"hooke": 2, "hertz": 2, "hertz/material": 3, "none": 0},
        "tangential": {
            "none": 0,
            "linear_nohistory": 2,
            "linear_history": 3,
            "mindlin": 3,
        },
        "damping": {
            "none": 0,
            "velocity": 0,
            "mass_velocity": 0,
            "viscoelastic": 0,
            "tsuji": 0,
            "coeff_restitution": 0,
        },
    }

    # Number of values each deform style takes after the style name, for a box
    # dimension and for a tilt factor respectively.
    _DEFORM_NARG: ClassVar[dict[str, int]] = {
        "final": 2, "delta": 2, "scale": 1, "vel": 1, "erate": 1, "trate": 1,
        "volume": 0, "wiggle": 2, "variable": 2, "pressure": 2, "pressure/mean": 2,
    }
    _DEFORM_TILT_NARG: ClassVar[dict[str, int]] = {
        "final": 1, "delta": 1, "vel": 1, "erate": 1, "trate": 1,
        "wiggle": 2, "variable": 2, "pressure": 2,
    }

    def _deform_spec(self, style: str, vals: list[str], tilt: bool) -> dict[str, Any]:
        """Turn a deform style and its raw arguments into an AxisSet keyword dict."""
        spec: dict[str, Any] = {"style": style}
        if style == "variable":
            # Two equal-style variable names, for the change and for its rate.
            spec["hstr"] = vals[0].removeprefix("v_")
            spec["hratestr"] = vals[1].removeprefix("v_")
            return spec

        nums = [float(self.evaluate_expression(v)) for v in vals]
        if style == "final":
            if tilt:
                spec["ftilt"] = nums[0]
            else:
                spec["flo"], spec["fhi"] = nums
        elif style == "delta":
            if tilt:
                spec["dtilt"] = nums[0]
            else:
                spec["dlo"], spec["dhi"] = nums
        elif style == "scale":
            spec["scale"] = nums[0]
        elif style == "vel":
            spec["vel"] = nums[0]
        elif style in ("erate", "trate"):
            spec["rate"] = nums[0]
        elif style == "wiggle":
            spec["amplitude"], spec["tperiod"] = nums
        elif style in ("pressure", "pressure/mean"):
            spec["ptarget"], spec["pgain"] = nums
        return spec

    def _parse_deform(self, args: list[str]) -> FixDeformPressure:
        """
        Parse `deform[/pressure] N <dim> <style> <values...> ... [keywords]`.

        Mirrors the FixDeform / FixDeformPressure argument grammar, including
        the tilt factors and the `box` layer.
        """
        if self.domain is None:
            raise ValueError("fix deform before the simulation box was defined")
        if not args:
            raise ValueError("fix deform needs an N argument")

        nevery = int(float(self.evaluate_expression(args[0])))
        axes: dict[str, dict[str, Any]] = {}
        box: dict[str, Any] | None = None
        couple = "none"
        max_rate = 0.0
        normalize_pressure = False
        vol_balance_p = False
        remap = "x"
        flip = True

        i = 1
        while i < len(args):
            token = args[i]
            if token in ("x", "y", "z", "xy", "xz", "yz"):
                tilt = token in ("xy", "xz", "yz")
                table = self._DEFORM_TILT_NARG if tilt else self._DEFORM_NARG
                style = args[i + 1]
                if style not in table:
                    raise ValueError(
                        f"Unsupported fix deform style {style!r} for {token}"
                    )
                n = table[style]
                axes[token] = self._deform_spec(style, args[i + 2 : i + 2 + n], tilt)
                i += 2 + n
            elif token == "box":
                style = args[i + 1]
                if style == "volume":
                    box = {"style": "volume"}
                    i += 2
                elif style == "pressure":
                    box = {
                        "style": "pressure",
                        "ptarget": float(self.evaluate_expression(args[i + 2])),
                        "pgain": float(self.evaluate_expression(args[i + 3])),
                    }
                    i += 4
                else:
                    raise ValueError(f"Unsupported fix deform box style {style!r}")
            elif token == "couple":
                couple = args[i + 1]
                i += 2
            elif token == "max/rate":
                max_rate = float(self.evaluate_expression(args[i + 1]))
                i += 2
            elif token == "normalize/pressure":
                normalize_pressure = args[i + 1].lower() == "yes"
                i += 2
            elif token == "vol/balance/p":
                vol_balance_p = args[i + 1].lower() == "yes"
                i += 2
            elif token == "remap":
                remap = args[i + 1]
                i += 2
            elif token == "flip":
                flip = args[i + 1].lower() == "yes"
                i += 2
            elif token == "units":
                # Only `box` units are meaningful without a lattice command.
                if args[i + 1].lower() != "box":
                    raise ValueError(
                        "fix deform units lattice needs a lattice command, which "
                        "taichimps does not have"
                    )
                i += 2
            else:
                raise ValueError(f"Unsupported fix deform keyword: {token!r}")

        return FixDeformPressure(
            domain=self.domain,
            axes=axes,
            box=box,
            couple=couple,
            max_rate=max_rate,
            normalize_pressure=normalize_pressure,
            vol_balance_p=vol_balance_p,
            remap=remap,
            nevery=nevery,
            flip=flip,
            var_eval=self.evaluate_variable,
        )

    def _parse_pair_coeff_granular(self, args: list[str]) -> PairGranular:
        """
        Parse `pair_coeff <i> <j> <normal> <coeffs...> [tangential ...] [damping ...]`.

        Follows PairGranular::coeff(): the normal model is mandatory and comes
        first without a keyword, the rest are introduced by their category
        keyword, and an unspecified damping model defaults to `viscoelastic`.
        Per-type coefficients are not supported yet, so the type selectors are
        accepted but every pair uses the same model.
        """
        if self.domain is None:
            raise ValueError("pair_coeff before the simulation box was defined")
        if len(args) < 3:
            raise ValueError("pair_coeff granular needs at least a normal model")

        def take(name: str, category: str, idx: int) -> tuple[list[float | None], int]:
            table = self._GRANULAR_NCOEFF[category]
            if name not in table:
                raise ValueError(
                    f"Unsupported pair granular {category} model {name!r}; "
                    f"implemented: {sorted(table)}"
                )
            n = table[name]
            if idx + n > len(args):
                raise ValueError(f"pair_coeff granular: {name} needs {n} coefficients")
            vals: list[float | None] = [
                None if tok == "NULL" else float(self.evaluate_expression(tok))
                for tok in args[idx : idx + n]
            ]
            return vals, idx + n

        i = 2
        normal_name = args[i]
        normal_coeffs, i = take(normal_name, "normal", i + 1)

        tangential_name = "none"
        tangential_coeffs: list[float | None] = []
        damping_name = "viscoelastic"
        limit_damping = False

        while i < len(args):
            key = args[i]
            if key == "tangential":
                tangential_name = args[i + 1]
                tangential_coeffs, i = take(tangential_name, "tangential", i + 2)
            elif key == "damping":
                damping_name = args[i + 1]
                _, i = take(damping_name, "damping", i + 2)
            elif key == "limit_damping":
                limit_damping = True
                i += 1
            else:
                raise ValueError(f"Unsupported pair_coeff granular keyword: {key!r}")

        return PairGranular(
            domain=self.domain,
            normal=normal_name,
            normal_coeffs=[c for c in normal_coeffs if c is not None],
            tangential=tangential_name,
            tangential_coeffs=tangential_coeffs,
            damping=damping_name,
            limit_damping=limit_damping,
        )

    def _parse_gran_settings(
        self, args: list[str]
    ) -> tuple[float, float, float, float, float, int, bool]:
        """
        Parse `Kn Kt gamma_n gamma_t xmu dampflag [limit_damping]`.

        Mirrors PairGranHookeHistory::settings(), including the NULL shorthands:
        a NULL tangential stiffness means Kn * 2/7 and a NULL tangential damping
        means gamma_n / 2 (these are not zero, which is what this used to read).
        """
        if len(args) < 6:
            raise ValueError(
                "pair_style gran/* needs Kn Kt gamma_n gamma_t xmu dampflag"
            )

        kn = float(self.evaluate_expression(args[0]))
        kt = kn * 2.0 / 7.0 if args[1] == "NULL" else float(self.evaluate_expression(args[1]))
        gamman = float(self.evaluate_expression(args[2]))
        gammat = 0.5 * gamman if args[3] == "NULL" else float(self.evaluate_expression(args[3]))
        xmu = float(self.evaluate_expression(args[4]))
        dampflag = int(float(self.evaluate_expression(args[5])))

        limit_damping = False
        if len(args) > 6:
            if args[6] != "limit_damping":
                raise ValueError(f"Unknown pair_style gran/* keyword: {args[6]}")
            limit_damping = True

        return kn, kt, gamman, gammat, xmu, dampflag, limit_damping

    def evaluate_variable(self, name: str) -> float:
        """
        Current value of a variable, re-evaluating an equal-style expression.

        `fix deform ... variable` needs the value as of this timestep, not as of
        the line that defined it.
        """
        name = name.removeprefix("v_")
        if name in self.variable_exprs:
            self.update_dynamic_variables()
            value = self.evaluate_expression(self.variable_exprs[name])
            if isinstance(value, str):
                raise ValueError(
                    f"variable {name!r} did not evaluate to a number: {value!r}"
                )
            self.variables[name] = value
            return float(value)
        stored = self.variables.get(name, 0.0)
        return float(stored) if isinstance(stored, (int, float)) else 0.0

    def evaluate_expression(self, expr_str: str) -> float | str:
        """Evaluate a mathematical expression or resolve variables."""
        # Substitute ${var}
        def sub_var(match: re.Match) -> str:
            var_name = match.group(1)
            val = self.variables.get(var_name, 0.0)
            return str(val)

        res = re.sub(r"\$\{([a-zA-Z0-9_]+)\}", sub_var, expr_str)
        # Substitute $var
        res = re.sub(r"\$([a-zA-Z0-9_]+)", sub_var, res)

        # Safe eval environment
        allowed_names = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "pi": math.pi,
            "abs": abs,
        }
        for k, v in self.variables.items():
            if isinstance(v, (int, float)):
                allowed_names[k] = v

        try:
            val = eval(res, {"__builtins__": None}, allowed_names)
            return float(val)
        except (ValueError, TypeError, SyntaxError, NameError, ZeroDivisionError):
            return res

    def parse_line(self, line: str) -> list[str]:
        """Strip comments and split line into arguments preserving quotes."""
        line = line.split("#")[0].strip()
        if not line:
            return []
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            tokens = line.split()

        expanded_tokens: list[str] = []
        is_var_def = len(tokens) >= 2 and tokens[0] == "variable"
        is_fix_print = len(tokens) >= 4 and tokens[0] == "fix" and tokens[3] == "print"
        for i, token in enumerate(tokens):
            # For fix print, don't expand string template at parse time!
            if is_fix_print and i == 5 or is_var_def and i < 3:
                expanded_tokens.append(token)
            else:
                # expand ${var}
                def sub_var(match: re.Match) -> str:
                    var_name = match.group(1)
                    val = self.variables.get(var_name, "")
                    return str(val)

                exp_t = re.sub(r"\$\{([a-zA-Z0-9_]+)\}", sub_var, token)
                exp_t = re.sub(r"\$([a-zA-Z0-9_]+)", sub_var, exp_t)
                expanded_tokens.append(exp_t)
        return expanded_tokens

    def load_script(self) -> None:
        """Reads and pre-processes input lines."""
        with open(self.script_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Handle line continuation with &
            while line.endswith("&"):
                line = line[:-1].strip()
                i += 1
                if i < len(lines):
                    line += " " + lines[i].strip()
            if line:
                self.commands.append(line)
            i += 1

    def execute(self) -> None:
        """Executes commands in sequence."""
        target_arch = self.arch if self.arch is not None else ti.cpu
        try:
            if self.default_fp is not None:
                ti.init(arch=target_arch, default_fp=self.default_fp)
            else:
                ti.init(arch=target_arch)
        except RuntimeError:
            pass

        for raw_cmd in self.commands:
            tokens = self.parse_line(raw_cmd)
            if not tokens:
                continue

            cmd = tokens[0]
            args = tokens[1:]

            if cmd == "variable":
                # variable <name> equal <expression>
                # variable <name> string <value>
                var_name = args[0]
                var_style = args[1]
                if var_style == "equal":
                    expr = "".join(args[2:])
                    self.variable_exprs[var_name] = expr
                    self.variables[var_name] = self.evaluate_expression(expr)
                elif var_style == "string":
                    self.variables[var_name] = args[2]
                else:
                    self.variables[var_name] = args[2]

            elif cmd == "boundary":
                # boundary p p p
                if len(args) >= 3 and self.domain:
                    self.domain.set_boundary((args[0], args[1], args[2]))

            elif cmd == "neigh_modify":
                # Parse neigh_modify arguments: delay <n>, every <n>, check <yes/no>
                if self.neighbor is None and self.domain is not None and self.atom is not None:
                    self.neighbor = NeighborList(domain=self.domain, max_atoms=self.atom.max_atoms)
                i_arg = 0
                while i_arg < len(args):
                    keyword = args[i_arg]
                    if keyword == "delay" and i_arg + 1 < len(args):
                        val_delay = int(self.evaluate_expression(args[i_arg + 1]))
                        if self.neighbor is not None:
                            self.neighbor.delay = val_delay
                        if self.simulation and self.simulation.neighbor:
                            self.simulation.neighbor.delay = val_delay
                        i_arg += 2
                    elif keyword == "every" and i_arg + 1 < len(args):
                        val_every = int(self.evaluate_expression(args[i_arg + 1]))
                        if self.neighbor is not None:
                            self.neighbor.every = val_every
                        if self.simulation and self.simulation.neighbor:
                            self.simulation.neighbor.every = val_every
                        i_arg += 2
                    elif keyword == "check" and i_arg + 1 < len(args):
                        val_check = args[i_arg + 1].lower() in ("yes", "1", "true")
                        if self.neighbor is not None:
                            self.neighbor.check = val_check
                        if self.simulation and self.simulation.neighbor:
                            self.simulation.neighbor.check = val_check
                        i_arg += 2
                    else:
                        i_arg += 1

            elif cmd == "read_data":
                data_file = self.workdir / args[0]
                data = read_data(data_file)
                self.domain = Domain(
                    boxlo=data.boxlo, boxhi=data.boxhi, tilt=data.tilt
                )
                self.atom = AtomSystem(max_atoms=data.natoms + 1000)
                self.atom.add_particles(
                    x=data.x,
                    radius=data.radius,
                    density=data.density,
                    v=data.v,
                    atom_type=data.atom_type,
                    tag=data.tag,
                )

            elif cmd == "create_box":
                # create_box <ntypes> <region-id>
                # If region exists, initialize domain with it
                reg_id = args[1]
                if reg_id in self.regions:
                    reg_data = self.regions[reg_id]
                    if isinstance(reg_data, tuple):
                        # A `prism` region carries tilt factors as a third entry.
                        b_lo, b_hi = reg_data[0], reg_data[1]
                        b_tilt = reg_data[2] if len(reg_data) > 2 else None
                        self.domain = Domain(b_lo, b_hi, tilt=b_tilt)
                        self.atom = AtomSystem(max_atoms=100000, float_type=self.default_fp or ti.f64)
                        # Add a default particle at box center
                        self.atom.add_particles(
                            x=[[0.5 * (b_lo[0] + b_hi[0]), 0.5 * (b_lo[1] + b_hi[1]), 0.5 * (b_lo[2] + b_hi[2])]],
                            v=[[0.0, 0.0, 0.0]],
                            omega=[[0.0, 0.0, 0.0]],
                            radius=[0.5],
                            density=1000.0,
                            atom_type=1,
                        )
                    elif isinstance(reg_data, dict):
                        # Cylinder or other region: approximate bounding box if lo/hi provided
                        c1, c2, r = reg_data["c1"], reg_data["c2"], reg_data["radius"]
                        lo = reg_data["lo"] if reg_data.get("lo") is not None else -r
                        hi = reg_data["hi"] if reg_data.get("hi") is not None else r
                        axis = reg_data.get("axis", "z")
                        if axis == "x":
                            b_lo = [lo, c1 - r, c2 - r]
                            b_hi = [hi, c1 + r, c2 + r]
                        elif axis == "y":
                            b_lo = [c1 - r, lo, c2 - r]
                            b_hi = [c1 + r, hi, c2 + r]
                        else:
                            b_lo = [c1 - r, c2 - r, lo]
                            b_hi = [c1 + r, c2 + r, hi]
                        self.domain = Domain(b_lo, b_hi)
                        self.atom = AtomSystem(max_atoms=100000)

            elif cmd == "region":
                # region <id> block <xlo> <xhi> <ylo> <yhi> <zlo> <zhi>
                # region <id> cylinder <dim> <c1> <c2> <radius> <lo> <hi> [side in|out]
                reg_id = args[0]
                style = args[1]
                if style == "prism" and len(args) >= 11:
                    bounds = [
                        float(self.evaluate_expression(args[2 + k])) for k in range(6)
                    ]
                    prism_tilt = [
                        float(self.evaluate_expression(args[8 + k])) for k in range(3)
                    ]
                    self.regions[reg_id] = (
                        [bounds[0], bounds[2], bounds[4]],
                        [bounds[1], bounds[3], bounds[5]],
                        prism_tilt,
                    )
                elif style == "block" and len(args) >= 8:
                    xlo = float(self.evaluate_expression(args[2]))
                    xhi = float(self.evaluate_expression(args[3]))
                    ylo = float(self.evaluate_expression(args[4]))
                    yhi = float(self.evaluate_expression(args[5]))
                    zlo = float(self.evaluate_expression(args[6]))
                    zhi = float(self.evaluate_expression(args[7]))
                    self.regions[reg_id] = ([xlo, ylo, zlo], [xhi, yhi, zhi])
                elif style == "cylinder" and len(args) >= 7:
                    axis_name = args[2].lower()
                    c1 = float(self.evaluate_expression(args[3]))
                    c2 = float(self.evaluate_expression(args[4]))
                    radius = float(self.evaluate_expression(args[5]))
                    lo = None if args[6] in ("INF", "NULL", "EDGE") else float(self.evaluate_expression(args[6]))
                    hi = None if len(args) <= 7 or args[7] in ("INF", "NULL", "EDGE") else float(self.evaluate_expression(args[7]))
                    side = "in"
                    idx = 8
                    while idx < len(args):
                        if args[idx] == "side" and idx + 1 < len(args):
                            side = args[idx + 1].lower()
                            idx += 2
                        else:
                            idx += 1
                    self.regions[reg_id] = {
                        "style": "cylinder",
                        "axis": axis_name,
                        "c1": c1,
                        "c2": c2,
                        "radius": radius,
                        "lo": lo,
                        "hi": hi,
                        "side": side,
                    }

            elif cmd == "unfix":
                fix_id = args[0]
                if fix_id in self.fixes:
                    f = self.fixes.pop(fix_id)
                    if self.simulation and f in self.simulation.fixes:
                        self.simulation.fixes.remove(f)

            elif cmd == "undump":
                dump_id = args[0]
                if dump_id in self.dumps and self.simulation:
                    d_writer = self.dumps.pop(dump_id)
                    self.simulation.dumps = [
                        (w, freq) for (w, freq) in self.simulation.dumps if w != d_writer
                    ]

            elif cmd == "timestep":
                self.dt = float(self.evaluate_expression(args[0]))
                if self.simulation:
                    self.simulation.dt = self.dt

            elif cmd == "thermo":
                self.thermo_freq = int(args[0])
                if self.simulation:
                    self.simulation.thermo_freq = self.thermo_freq

            elif cmd == "thermo_style":
                if args[0] == "custom":
                    self.custom_thermo_keys = args[1:]

            elif cmd == "pair_style":
                style_name = args[0]
                if self.domain is None:
                    continue
                if style_name in ("gran/hertz/history", "gran/hooke/history", "gran/hooke"):
                    kn, kt, gamman, gammat, xmu, dampflag, limit_damping = (
                        self._parse_gran_settings(args[1:])
                    )
                    if style_name == "gran/hertz/history":
                        self.pair_style = GranHertzHistory(
                            domain=self.domain, kn=kn, kt=kt, gamman=gamman,
                            gammat=gammat, xmu=xmu, dampflag=dampflag,
                            limit_damping=limit_damping,
                        )
                    elif style_name == "gran/hooke/history":
                        self.pair_style = GranHookeHistory(
                            domain=self.domain, kn=kn, kt=kt, gamman=gamman,
                            gammat=gammat, xmu=xmu, dampflag=dampflag,
                            limit_damping=limit_damping,
                        )
                    else:
                        self.pair_style = GranHooke(
                            domain=self.domain, kn=kn, kt=kt, gamman=gamman,
                            gammat=gammat, xmu=xmu, dampflag=dampflag,
                            limit_damping=limit_damping,
                        )
                elif style_name == "granular":
                    # All model choices live in pair_coeff for this style, so
                    # the pair style itself is only built once that arrives.
                    self.pair_style = None

            elif cmd == "pair_coeff":
                if self.domain is None:
                    continue
                self.pair_style = self._parse_pair_coeff_granular(args)
                if self.simulation is not None:
                    self.simulation.set_pair_style(self.pair_style)

            elif cmd == "fix":
                fix_id = args[0]
                # group = args[1]
                fix_style = args[2]
                fix_args = args[3:]

                if self.simulation is None and self.atom is not None and self.domain is not None:
                    self.simulation = Simulation(
                        domain=self.domain,
                        atom=self.atom,
                        pair=self.pair_style,
                        dt=self.dt,
                        thermo_freq=self.thermo_freq,
                    )

                # A plane wall/gran command can expand into two fixes (lo and hi).
                fix_inst: Fix | list[Fix] | None = None
                if fix_style == "nve/sphere":
                    if self.domain:
                        fix_inst = FixNVESphere(domain=self.domain)
                        self.fixes[fix_id] = fix_inst
                        if self.simulation:
                            self.simulation.add_fix(fix_inst)

                elif fix_style == "damping/cundall":
                    gamma_lin = float(self.evaluate_expression(fix_args[0]))
                    gamma_ang = float(self.evaluate_expression(fix_args[1]))
                    if self.domain:
                        fix_inst = FixDampingCundall(
                            domain=self.domain,
                            gamma_lin=gamma_lin,
                            gamma_ang=gamma_ang,
                        )
                        self.fixes[fix_id] = fix_inst
                        if self.simulation:
                            self.simulation.add_fix(fix_inst)

                elif fix_style in ("deform/pressure", "deform"):
                    if self.domain:
                        fix_inst = self._parse_deform(fix_args)
                        self.fixes[fix_id] = fix_inst
                        if self.simulation:
                            self.simulation.add_fix(fix_inst)

                elif fix_style == "gravity":
                    mag = float(self.evaluate_expression(fix_args[0]))
                    if self.domain:
                        fix_inst = FixGravity(domain=self.domain, magnitude=mag)
                        self.fixes[fix_id] = fix_inst
                        if self.simulation:
                            self.simulation.add_fix(fix_inst)

                elif fix_style == "print":
                    # fix <id> <group> print <nevery> <template_str> [file <path>] [screen yes/no] [title <str>]
                    nevery = int(fix_args[0])
                    template_str = fix_args[1]
                    filepath = None
                    title = None
                    screen = False
                    idx = 2
                    while idx < len(fix_args):
                        opt = fix_args[idx]
                        if opt == "file" and idx + 1 < len(fix_args):
                            filepath = self.workdir / fix_args[idx + 1]
                            idx += 2
                        elif opt == "title" and idx + 1 < len(fix_args):
                            title = fix_args[idx + 1]
                            idx += 2
                        elif opt == "screen" and idx + 1 < len(fix_args):
                            screen = (fix_args[idx + 1].lower() == "yes")
                            idx += 2
                        else:
                            idx += 1

                    fix_inst = FixPrint(
                        domain=self.domain,
                        nevery=nevery,
                        template_str=template_str,
                        eval_fn=self.evaluate_string_template,
                        filepath=filepath,
                        title=title,
                        screen=screen,
                    )
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style == "viscous" and self.domain:
                    gamma = float(self.evaluate_expression(fix_args[0]))
                    fix_inst = FixViscousSphere(domain=self.domain, gamma=gamma)
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style == "drag" and self.domain:
                    f_drag = float(self.evaluate_expression(fix_args[0]))
                    fix_inst = FixDrag(domain=self.domain, f_drag=f_drag)
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style == "freeze" and self.domain:
                    fix_inst = FixFreeze(domain=self.domain)
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style in ("wave", "wave/source") and self.domain:
                    # fix <id> <group> wave [axis <x|y|z>] [f0 <float>] [t0 <float>] [amplitude <float>] [waveform <ricker|sine|pulse>] [mode <force|velocity>] [region <id>]
                    axis = "z"
                    f0 = 100.0
                    t0 = None
                    amplitude = 1.0
                    waveform = "ricker"
                    mode = "force"
                    region = None
                    i = 0
                    while i < len(fix_args):
                        opt = fix_args[i].lower()
                        if opt in ("axis", "x", "y", "z"):
                            if opt in ("x", "y", "z"):
                                axis = opt
                                i += 1
                            elif i + 1 < len(fix_args):
                                axis = fix_args[i + 1]
                                i += 2
                        elif opt in ("ricker", "sine", "pulse", "gaussian"):
                            waveform = opt
                            i += 1
                        elif opt in ("f0", "freq") and i + 1 < len(fix_args):
                            f0 = float(self.evaluate_expression(fix_args[i + 1]))
                            i += 2
                        elif opt == "t0" and i + 1 < len(fix_args):
                            t0 = float(self.evaluate_expression(fix_args[i + 1]))
                            i += 2
                        elif opt in ("amplitude", "amp") and i + 1 < len(fix_args):
                            amplitude = float(self.evaluate_expression(fix_args[i + 1]))
                            i += 2
                        elif opt == "waveform" and i + 1 < len(fix_args):
                            waveform = fix_args[i + 1].lower()
                            i += 2
                        elif opt == "mode" and i + 1 < len(fix_args):
                            mode = fix_args[i + 1].lower()
                            i += 2
                        elif opt == "region" and i + 1 < len(fix_args):
                            reg_id = fix_args[i + 1]
                            if reg_id in self.regions:
                                reg_val = self.regions[reg_id]
                                if isinstance(reg_val, tuple):
                                    region = reg_val[0] + reg_val[1]
                            i += 2
                        else:
                            i += 1

                    fix_inst = FixWave(
                        domain=self.domain,
                        float_type=self.default_fp or ti.f64,
                        axis=axis,
                        waveform=waveform,
                        f0=f0,
                        t0=t0,
                        amplitude=amplitude,
                        mode=mode,
                        region=region,
                    )
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style in ("sponge", "winsponge", "sponge/boundary") and self.domain:
                    # fix <id> <group> sponge [thick|thickness <float>] [eta|eta_max <float>] [window <polynomial|hann>] [power <float>]
                    thickness = 1.0
                    eta_max = 50.0
                    window = "polynomial"
                    power = 2.0
                    boundaries = None
                    idx = 0
                    while idx < len(fix_args):
                        opt = fix_args[idx].lower()
                        if opt in ("thick", "thickness") and idx + 1 < len(fix_args):
                            thickness = float(self.evaluate_expression(fix_args[idx + 1]))
                            idx += 2
                        elif opt in ("eta", "eta_max") and idx + 1 < len(fix_args):
                            eta_max = float(self.evaluate_expression(fix_args[idx + 1]))
                            idx += 2
                        elif opt == "window" and idx + 1 < len(fix_args):
                            window = fix_args[idx + 1].lower()
                            idx += 2
                        elif opt == "power" and idx + 1 < len(fix_args):
                            power = float(self.evaluate_expression(fix_args[idx + 1]))
                            idx += 2
                        elif opt in ("boundaries", "faces") and idx + 1 < len(fix_args):
                            boundaries = fix_args[idx + 1].split(",")
                            idx += 2
                        else:
                            try:
                                thickness = float(self.evaluate_expression(opt))
                                idx += 1
                            except (ValueError, TypeError):
                                idx += 1

                    fix_inst = WinSponge(
                        domain=self.domain,
                        float_type=self.default_fp or ti.f64,
                        thickness=thickness,
                        eta_max=eta_max,
                        power=power,
                        window=window,
                        boundaries=boundaries,
                    )
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style in ("probe", "ave/probe", "wave/probe") and self.domain:
                    # fix <id> <group> probe <nevery> [points x1,y1,z1;x2,y2,z2] [radius <r>] [file <path>]
                    # OR fix <id> <group> probe <nevery> <x1> <y1> <z1> [radius <r>] [file <path>]
                    nevery = int(self.evaluate_expression(fix_args[0]))
                    radius = 1.0
                    points_list = []
                    filepath = None
                    i = 1
                    # Check if next 3 args are raw coordinates: x y z
                    if len(fix_args) >= 4:
                        try:
                            x0 = float(self.evaluate_expression(fix_args[1]))
                            y0 = float(self.evaluate_expression(fix_args[2]))
                            z0 = float(self.evaluate_expression(fix_args[3]))
                            points_list.append([x0, y0, z0])
                            i = 4
                        except (ValueError, TypeError):
                            pass
                    while i < len(fix_args):
                        opt = fix_args[i].lower()
                        if opt in ("radius", "r") and i + 1 < len(fix_args):
                            radius = float(self.evaluate_expression(fix_args[i + 1]))
                            i += 2
                        elif opt in ("file", "out") and i + 1 < len(fix_args):
                            filepath = self.workdir / fix_args[i + 1]
                            i += 2
                        elif opt in ("points", "pts") and i + 1 < len(fix_args):
                            pts_str = fix_args[i + 1]
                            for pstr in pts_str.split(";"):
                                coords = [float(c) for c in pstr.split(",")]
                                if len(coords) == 3:
                                    points_list.append(coords)
                            i += 2
                        elif opt in ("every", "nevery") and i + 1 < len(fix_args):
                            nevery = int(self.evaluate_expression(fix_args[i + 1]))
                            i += 2
                        else:
                            i += 1

                    if not points_list and len(fix_args) >= 4:
                        # Check if coordinates were passed directly: probe <nevery> <x> <y> <z> ...
                        try:
                            px = float(self.evaluate_expression(fix_args[1]))
                            py = float(self.evaluate_expression(fix_args[2]))
                            pz = float(self.evaluate_expression(fix_args[3]))
                            points_list = [[px, py, pz]]
                        except (ValueError, TypeError):
                            pass

                    if not points_list:
                        # Default center point
                        cx = 0.5 * (self.domain.boxlo[0] + self.domain.boxhi[0])
                        cy = 0.5 * (self.domain.boxlo[1] + self.domain.boxhi[1])
                        cz = 0.5 * (self.domain.boxlo[2] + self.domain.boxhi[2])
                        points_list = [[cx, cy, cz]]

                    fix_inst = FixProbe(
                        domain=self.domain,
                        float_type=self.default_fp or ti.f64,
                        points=points_list,
                        radius=radius,
                        nevery=nevery,
                        file=filepath,
                    )
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style in ("triaxial", "triax/servo") and self.domain:
                    # fix <id> <group> triaxial [target_stress sx sy sz] [strain_rate ex ey ez] [stress_mask mx my mz]
                    target_stress = [-100.0e3, -100.0e3, -100.0e3]
                    strain_rate = [0.0, 0.0, 0.0]
                    stress_mask = [1, 1, 1]
                    stress_damping = 0.5
                    max_velocity = 0.1
                    i = 0
                    while i < len(fix_args):
                        opt = fix_args[i].lower()
                        if opt == "target_stress" and i + 3 < len(fix_args):
                            target_stress = [
                                float(self.evaluate_expression(fix_args[i + 1])),
                                float(self.evaluate_expression(fix_args[i + 2])),
                                float(self.evaluate_expression(fix_args[i + 3])),
                            ]
                            i += 4
                        elif opt == "strain_rate" and i + 3 < len(fix_args):
                            strain_rate = [
                                float(self.evaluate_expression(fix_args[i + 1])),
                                float(self.evaluate_expression(fix_args[i + 2])),
                                float(self.evaluate_expression(fix_args[i + 3])),
                            ]
                            i += 4
                        elif opt == "stress_mask" and i + 3 < len(fix_args):
                            stress_mask = [
                                int(self.evaluate_expression(fix_args[i + 1])),
                                int(self.evaluate_expression(fix_args[i + 2])),
                                int(self.evaluate_expression(fix_args[i + 3])),
                            ]
                            i += 4
                        elif opt == "stress_damping" and i + 1 < len(fix_args):
                            stress_damping = float(self.evaluate_expression(fix_args[i + 1]))
                            i += 2
                        elif opt == "max_velocity" and i + 1 < len(fix_args):
                            max_velocity = float(self.evaluate_expression(fix_args[i + 1]))
                            i += 2
                        else:
                            i += 1

                    fix_inst = FixTriaxial(
                        domain=self.domain,
                        float_type=self.default_fp or ti.f64,
                        target_stress=target_stress,
                        strain_rate=strain_rate,
                        stress_mask=stress_mask,
                        stress_damping=stress_damping,
                        max_velocity=max_velocity,
                    )
                    self.fixes[fix_id] = fix_inst
                    if self.simulation:
                        self.simulation.add_fix(fix_inst)

                elif fix_style.startswith("wall/gran") and self.domain:
                    # Parse wall/gran and wall/gran/region
                    # fix <id> <group> wall/gran/region <fstyle> ... region <reg_id>
                    # fix <id> <group> wall/gran <fstyle> ... <wallstyle> ...
                    fstyle = "hooke"
                    kn = 1e5
                    gamman = 0.0
                    kt = None
                    gammat = None
                    xmu = 0.0
                    dampflag = 1
                    wallstyle = None
                    plane_axis = 2
                    plane_bounds: list[float | None] = [None, None]
                    region_id = None
                    axis_str = "z"
                    c1 = 0.0
                    c2 = 0.0
                    radius = 1.0
                    side_val = "in"

                    # Parse parameters
                    # First argument after fix_style can be fstyle or granular or wallstyle
                    i = 0
                    if i < len(fix_args) and fix_args[i].lower() in ("hooke", "hooke/history", "hertz", "hertz/history", "granular"):
                        fstyle = fix_args[i].lower()
                        i += 1
                        # If numeric args follow: Kn, Kt, gamma_n, gamma_t, xmu, dampflag
                        num_params: list[float | None] = []
                        while i < len(fix_args) and fix_args[i].lower() not in (
                            "xplane", "yplane", "zplane", "cylinder", "zcylinder", "region"
                        ):
                            val_str = fix_args[i]
                            if val_str.upper() == "NULL":
                                num_params.append(None)
                            else:
                                try:
                                    num_params.append(float(self.evaluate_expression(val_str)))
                                except ValueError:
                                    break
                            i += 1

                        if len(num_params) >= 1 and num_params[0] is not None:
                            kn = num_params[0]
                        if len(num_params) >= 2 and num_params[1] is not None:
                            kt = num_params[1]
                        if len(num_params) >= 3 and num_params[2] is not None:
                            gamman = num_params[2]
                        if len(num_params) >= 4 and num_params[3] is not None:
                            gammat = num_params[3]
                        if len(num_params) >= 5 and num_params[4] is not None:
                            xmu = num_params[4]
                        if len(num_params) >= 6 and num_params[5] is not None:
                            dampflag = int(num_params[5])

                    # Check for wallstyle or region in remaining tokens
                    while i < len(fix_args):
                        token = fix_args[i].lower()
                        if token == "region" and i + 1 < len(fix_args):
                            region_id = fix_args[i + 1]
                            i += 2
                        elif token in ("xplane", "yplane", "zplane"):
                            wallstyle = token
                            # plane params: lo hi, either of which may be NULL
                            plane_axis = "xyz".index(token[0])
                            for slot in range(2):
                                if i + 1 + slot >= len(fix_args):
                                    break
                                tok = fix_args[i + 1 + slot]
                                if tok.upper() != "NULL":
                                    plane_bounds[slot] = float(
                                        self.evaluate_expression(tok)
                                    )
                            i += 3
                        elif token == "cylinder":
                            wallstyle = "cylinder"
                            # cylinder <axis> <c1> <c2> <radius>
                            if i + 4 < len(fix_args):
                                axis_str = fix_args[i + 1].lower()
                                c1 = float(self.evaluate_expression(fix_args[i + 2]))
                                c2 = float(self.evaluate_expression(fix_args[i + 3]))
                                radius = float(self.evaluate_expression(fix_args[i + 4]))
                                i += 5
                            else:
                                i += 1
                        elif token == "zcylinder":
                            wallstyle = "cylinder"
                            axis_str = "z"
                            # zcylinder <radius> [c1 c2]
                            if i + 1 < len(fix_args):
                                radius = float(self.evaluate_expression(fix_args[i + 1]))
                                i += 2
                                if i + 1 < len(fix_args) and not fix_args[i].isalpha():
                                    c1 = float(self.evaluate_expression(fix_args[i]))
                                    c2 = float(self.evaluate_expression(fix_args[i + 1]))
                                    i += 2
                            else:
                                i += 1
                        else:
                            i += 1

                    if region_id and region_id in self.regions:
                        reg = self.regions[region_id]
                        if isinstance(reg, dict) and reg.get("style") == "cylinder":
                            fix_inst = FixWallGranRegion(
                                domain=self.domain,
                                axis=reg["axis"],
                                c1=reg["c1"],
                                c2=reg["c2"],
                                radius=reg["radius"],
                                side="out" if str(reg.get("side", "in")).lower() == "out" else "in",
                                axis_lo=reg.get("lo"),
                                axis_hi=reg.get("hi"),
                                fstyle=fstyle,
                                kn=kn,
                                gamman=gamman,
                                kt=kt if kt is not None else 0.0,
                                gammat=gammat,
                                xmu=xmu,
                                dampflag=dampflag,
                                max_atoms=(
                                    self.atom.max_atoms if self.atom else 100000
                                ),
                            )
                        else:
                            # Default fallback or planar region
                            fix_inst = FixWallGran(
                                domain=self.domain,
                                wall_axis=2,
                                wall_side=-1,
                                wall_coord=0.0,
                                kn=kn,
                                gamman=gamman,
                                kt=kt if kt is not None else 0.0,
                                gammat=gammat if gammat is not None else 0.0,
                                xmu=xmu,
                            )
                    elif wallstyle == "cylinder":
                        side_str: Literal["in", "out"] = "out" if side_val == "out" or side_val == 1 else "in"
                        fix_inst = FixWallGranRegion(
                            domain=self.domain,
                            axis=axis_str,
                            c1=c1,
                            c2=c2,
                            radius=radius,
                            side=side_str,
                            fstyle=fstyle,
                            kn=kn,
                            gamman=gamman,
                            kt=kt,
                            gammat=gammat,
                            xmu=xmu,
                            dampflag=dampflag,
                            max_atoms=(
                                self.atom.max_atoms if self.atom else 100000
                            ),
                        )
                    else:
                        # Plane wall(s). LAMMPS takes `<dim>plane lo hi`, either
                        # of which may be NULL, and one fix serves both; here a
                        # separate instance per wall adds up to the same forces.
                        # NULL shorthands follow the pair styles: Kt = Kn*2/7,
                        # gamma_t = gamma_n/2.
                        kt_val = kt if kt is not None else kn * 2.0 / 7.0
                        gammat_val = gammat if gammat is not None else 0.5 * gamman
                        use_history = fstyle.endswith("history")
                        if plane_bounds == [None, None]:
                            plane_bounds = [
                                float(self.domain.boxlo[plane_axis]),
                                None,
                            ]
                        walls = []
                        for slot, coord in enumerate(plane_bounds):
                            if coord is None:
                                continue
                            walls.append(
                                FixWallGran(
                                    domain=self.domain,
                                    wall_axis=plane_axis,
                                    wall_side=-1 if slot == 0 else 1,
                                    wall_coord=coord,
                                    kn=kn,
                                    gamman=gamman,
                                    kt=kt_val,
                                    gammat=gammat_val,
                                    xmu=xmu,
                                    dampflag=dampflag,
                                    history=use_history,
                                    max_atoms=(
                                        self.atom.max_atoms if self.atom else 100000
                                    ),
                                )
                            )
                        fix_inst = walls[0] if len(walls) == 1 else list(walls)

                    self.fixes[fix_id] = fix_inst
                    if self.simulation and fix_inst is not None:
                        for one in fix_inst if isinstance(fix_inst, list) else [fix_inst]:
                            self.simulation.add_fix(one)

            elif cmd == "unfix":
                fix_id = args[0]
                if fix_id in self.fixes:
                    inst = self.fixes.pop(fix_id)
                    # A plane wall/gran command can expand into two fixes.
                    for one in inst if isinstance(inst, list) else [inst]:
                        if self.simulation and one in self.simulation.fixes:
                            self.simulation.fixes.remove(one)

            elif cmd == "dump":
                # dump <id> <group> custom <freq> <file> <args...>
                # id = args[0]
                freq = int(args[2])
                dump_file = self.workdir / args[3]
                writer = DumpWriter(filepath=dump_file)
                if self.simulation:
                    self.simulation.add_dump(writer, freq=freq)

            elif cmd == "run":
                nsteps = int(args[0])
                if self.simulation is None and self.atom and self.domain:
                    self.simulation = Simulation(
                        domain=self.domain,
                        atom=self.atom,
                        pair=self.pair_style,
                        dt=self.dt,
                        thermo_freq=self.thermo_freq,
                    )
                    for f in self.fixes.values():
                        self.simulation.add_fix(f)

                if self.simulation:
                    # Update pair if changed
                    if self.pair_style and self.simulation.pair_style != self.pair_style:
                        self.simulation.pair_style = self.pair_style
                    for f in self.fixes.values():
                        if f not in self.simulation.fixes:
                            self.simulation.add_fix(f)
                    self.simulation.run(nsteps)


LammpsInputParser = LAMMPSInputParser


def parse_and_run(script_path: str | Path) -> Simulation | None:
    """Parse and run a LAMMPS script."""
    parser = LAMMPSInputParser(script_path)
    parser.execute()
    return parser.simulation

