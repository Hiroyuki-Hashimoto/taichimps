"""LAMMPS script parser module compatible with LAMMPS script syntax."""
from taichimps.input import LammpsInputParser, parse_and_run

ScriptParser = LammpsInputParser

__all__ = ["LammpsInputParser", "ScriptParser", "parse_and_run"]
