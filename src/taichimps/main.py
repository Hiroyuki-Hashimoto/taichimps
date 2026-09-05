"""Entry point for taichimps CLI."""

import argparse
import sys

import taichi as ti

from taichimps.input import LAMMPSInputParser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="taichimps: High-Performance GPU/CPU Granular DEM Engine (Taichi-Lang)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("script", nargs="?", help="Path to LAMMPS input script to execute")
    parser.add_argument("-in", "--input", dest="in_script", help="LAMMPS-style -in <script> option")
    parser.add_argument(
        "--arch",
        default="vulkan",
        choices=["vulkan", "cuda", "cpu", "opengl"],
        help="Compute architecture backend (defaults to vulkan for high-speed cross-platform GPU execution)",
    )
    parser.add_argument(
        "--fp",
        "--dtype",
        dest="fp",
        default="f32",
        choices=["f32", "f64"],
        help="Default floating-point precision (f32 or f64)",
    )
    args = parser.parse_args()

    script_path = args.script or args.in_script

    if not script_path:
        print("taichimps: High-Performance GPU/CPU Granular DEM Engine")
        print("Usage: taichimps <in.script> [--arch vulkan|cuda|cpu] [--fp f32|f64]")
        print("       taichimps -in <in.script> [--arch vulkan|cuda|cpu]")
        sys.exit(0)

    arch_map = {
        "vulkan": ti.vulkan,
        "cuda": ti.cuda,
        "cpu": ti.cpu,
        "opengl": ti.opengl,
    }
    selected_arch = arch_map.get(args.arch, ti.vulkan)
    default_fp = ti.f32 if args.fp == "f32" else ti.f64

    try:
        ti.init(arch=selected_arch, default_fp=default_fp)
    except Exception as e:
        print(f"[taichimps WARNING] Failed to initialize backend {args.arch} ({e}), falling back to CPU...")
        selected_arch = ti.cpu
        ti.init(arch=ti.cpu, default_fp=default_fp)

    lmp_parser = LAMMPSInputParser(script_path, default_fp=default_fp, arch=selected_arch)
    lmp_parser.execute()


if __name__ == "__main__":
    main()
