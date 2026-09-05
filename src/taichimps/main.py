"""Entry point for taichimps CLI."""

import argparse
import sys

import taichi as ti

from taichimps.input import LAMMPSInputParser


def main() -> None:
    parser = argparse.ArgumentParser(description="taichimps: High-Performance Granular DEM Engine")
    parser.add_argument("script", nargs="?", help="Path to LAMMPS input script to execute")
    parser.add_argument("--arch", default="cpu", choices=["cpu", "cuda", "vulkan", "opengl"], help="Compute backend")
    args = parser.parse_args()

    if not args.script:
        print("taichimps: High-Performance Granular DEM Engine")
        print("Usage: taichimps <in.script> [--arch cpu|cuda|vulkan]")
        sys.exit(0)

    arch_map = {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "opengl": ti.opengl,
    }
    ti.init(arch=arch_map.get(args.arch, ti.cpu))

    lmp_parser = LAMMPSInputParser(args.script)
    lmp_parser.execute()


if __name__ == "__main__":
    main()
