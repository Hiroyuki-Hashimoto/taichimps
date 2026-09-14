"""
Dump writer for simulation outputs: LAMMPS dump format (.dump) and VTK (.vtp / .vtk).
Reference: LAMMPS src/dump_custom.cpp
"""

from pathlib import Path

from taichimps.atom import AtomSystem
from taichimps.domain import Domain


class DumpWriter:
    """Outputs particle configurations to disk."""

    def __init__(self, filepath: str | Path | None = None) -> None:
        self.filepath = filepath

    def write_timestep(
        self,
        timestep: int,
        domain: Domain,
        atom: AtomSystem,
        append: bool = True,
    ) -> None:
        if self.filepath is None:
            raise ValueError("filepath not configured for DumpWriter")
        self.write_lammps_dump(
            filepath=self.filepath,
            timestep=timestep,
            domain=domain,
            atom=atom,
            append=append,
        )

    def write_dump(
        self,
        timestep: int,
        domain: Domain,
        atom: AtomSystem,
        append: bool = True,
    ) -> None:
        """Alias for write_timestep."""
        self.write_timestep(timestep, domain, atom, append=append)

    @staticmethod
    def write_lammps_dump(
        filepath: str | Path,
        timestep: int,
        domain: Domain,
        atom: AtomSystem,
        append: bool = True,
    ) -> None:
        """
        Write or append a snapshot in LAMMPS custom dump format:
        ITEM: TIMESTEP
        ITEM: NUMBER OF ATOMS
        ITEM: BOX BOUNDS
        ITEM: ATOMS id type x y z vx vy vz omegax omegay omegaz radius
        """
        mode = "a" if append else "w"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Pull data from Taichi fields to numpy
        nlocal = atom.nlocal
        pos = atom.x.to_numpy()[:nlocal]
        vel = atom.v.to_numpy()[:nlocal]
        omega = atom.omega.to_numpy()[:nlocal]
        radius = atom.radius.to_numpy()[:nlocal]
        tag = atom.tag.to_numpy()[:nlocal]
        type_ = atom.atom_type.to_numpy()[:nlocal]

        triclinic = domain.triclinic
        # A triclinic dump reports the axis-aligned bounding box of the tilted
        # cell plus a tilt factor per line, which is what LAMMPS writes.
        if triclinic:
            lo, hi = domain.bounding_box()
        else:
            lo, hi = domain.boxlo, domain.boxhi
        pbc_flags = [
            "pp" if p else "ff"
            for p in (domain.pbc_x, domain.pbc_y, domain.pbc_z)
        ]

        with open(path, mode, encoding="utf-8") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{nlocal}\n")
            if triclinic:
                xy, xz, yz = (float(t) for t in domain.tilt)
                f.write(
                    "ITEM: BOX BOUNDS xy xz yz "
                    f"{pbc_flags[0]} {pbc_flags[1]} {pbc_flags[2]}\n"
                )
                f.write(f"{lo[0]:.8e} {hi[0]:.8e} {xy:.8e}\n")
                f.write(f"{lo[1]:.8e} {hi[1]:.8e} {xz:.8e}\n")
                f.write(f"{lo[2]:.8e} {hi[2]:.8e} {yz:.8e}\n")
            else:
                f.write(
                    f"ITEM: BOX BOUNDS {pbc_flags[0]} {pbc_flags[1]} {pbc_flags[2]}\n"
                )
                f.write(f"{lo[0]:.8e} {hi[0]:.8e}\n")
                f.write(f"{lo[1]:.8e} {hi[1]:.8e}\n")
                f.write(f"{lo[2]:.8e} {hi[2]:.8e}\n")
            f.write("ITEM: ATOMS id type x y z vx vy vz omegax omegay omegaz radius\n")
            f.writelines(f"{tag[i]} {type_[i]} "
                    f"{pos[i, 0]:.6e} {pos[i, 1]:.6e} {pos[i, 2]:.6e} "
                    f"{vel[i, 0]:.6e} {vel[i, 1]:.6e} {vel[i, 2]:.6e} "
                    f"{omega[i, 0]:.6e} {omega[i, 1]:.6e} {omega[i, 2]:.6e} "
                    f"{radius[i]:.6e}\n" for i in range(nlocal))
