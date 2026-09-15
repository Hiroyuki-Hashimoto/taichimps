"""
LAMMPS binary restart files.

Reference: LAMMPS src/write_restart.cpp, read_restart.cpp, lmprestart.h,
           atom_vec.cpp (pack_restart/unpack_restart),
           atom_vec_sphere.cpp (fields_restart),
           fix_neigh_history.cpp (pack_restart)
License: GPL v2 compatible / MIT reimplementation

The format is a flat sequence of little-endian records:

    "LammpS RestartT\\0"      magic string, NUL terminated
    int endian                 0x0001
    int format revision        3

    header      (int flag, payload) pairs, terminated by flag = -1
    groups      int ngroup, then a length-prefixed name each
    type arrays (int flag, payload) pairs, terminated by flag = -1
    force field (int flag, payload) pairs, terminated by flag = -1
    fix state   two counted lists: fixes with global state, then with per-atom
    file layout (int flag, payload) pairs, terminated by flag = -1

    per-processor chunks, each:
        int PERPROC, int ndoubles, ndoubles * double

A chunk holds one record per atom, back to back.  The first double of a record
is its own length, so records can be walked without knowing what produced them.
For `atom_style sphere` a record is

    [m, x, y, z, tag, type, mask, image, vx, vy, vz,
     radius, rmass, omegax, omegay, omegaz, <fix data...>]

where tag/type/mask/image are 64-bit integers reinterpreted as doubles (the
LAMMPS `ubuf` union), and the fix data is whatever fixes with per-atom restart
state appended -- for granular runs that is FixNeighHistory, carrying the
tangential contact history.

Only what taichimps can act on is interpreted; unknown header flags are read
past rather than rejected, so a file written by a LAMMPS build with extra
packages still loads.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"LammpS RestartT\x00"
# ReadRestart::header() puts the VERSION string through utils::date2num(), so
# it has to be a LAMMPS-style release date, not a free-form name.  This is the
# release whose restart layout this module was written against.
VERSION_STRING = "2 Sep 2026"
ENDIAN = 0x0001
FORMAT_REVISION = 3

# lmprestart.h, in order.
(
    VERSION, SMALLINT, TAGINT, BIGINT,
    UNITS, NTIMESTEP, DIMENSION, NPROCS, PROCGRID,
    NEWTON_PAIR, NEWTON_BOND,
    XPERIODIC, YPERIODIC, ZPERIODIC, BOUNDARY,
    ATOM_STYLE, NATOMS, NTYPES,
    NBONDS, NBONDTYPES, BOND_PER_ATOM,
    NANGLES, NANGLETYPES, ANGLE_PER_ATOM,
    NDIHEDRALS, NDIHEDRALTYPES, DIHEDRAL_PER_ATOM,
    NIMPROPERS, NIMPROPERTYPES, IMPROPER_PER_ATOM,
    TRICLINIC, BOXLO, BOXHI, XY, XZ, YZ,
    SPECIAL_LJ, SPECIAL_COUL,
    MASS, PAIR, BOND, ANGLE, DIHEDRAL, IMPROPER,
    MULTIPROC, MPIIO, PROCSPERFILE, PERPROC,
    IMAGEINT, BOUNDMIN, TIMESTEP,
    ATOM_ID, ATOM_MAP_STYLE, ATOM_MAP_USER, ATOM_SORTFREQ, ATOM_SORTBIN,
    COMM_MODE, COMM_CUTOFF, COMM_VEL, NO_PAIR,
    EXTRA_BOND_PER_ATOM, EXTRA_ANGLE_PER_ATOM, EXTRA_DIHEDRAL_PER_ATOM,
    EXTRA_IMPROPER_PER_ATOM, EXTRA_SPECIAL_PER_ATOM, ATOM_MAXSPECIAL,
    NELLIPSOIDS, NLINES, NTRIS, NBODIES, ATIME, ATIMESTEP, LABELMAP,
    TRICLINIC_GENERAL, ROTATE_G2R, ATOM_MAXEXCHANGE,
) = range(76)

# How each header flag's payload is stored. Anything absent here is skipped by
# reading its declared length, so an unfamiliar flag is not fatal.
_INT_FLAGS = frozenset({
    SMALLINT, TAGINT, BIGINT, IMAGEINT, DIMENSION, NPROCS,
    NEWTON_PAIR, NEWTON_BOND, XPERIODIC, YPERIODIC, ZPERIODIC,
    NTYPES, NBONDTYPES, BOND_PER_ATOM, NANGLETYPES, ANGLE_PER_ATOM,
    NDIHEDRALTYPES, DIHEDRAL_PER_ATOM, NIMPROPERTYPES, IMPROPER_PER_ATOM,
    TRICLINIC, TRICLINIC_GENERAL, MULTIPROC, MPIIO, PROCSPERFILE,
    ATOM_ID, ATOM_MAP_STYLE, ATOM_MAP_USER, ATOM_SORTFREQ,
    COMM_MODE, COMM_VEL, NO_PAIR,
    EXTRA_BOND_PER_ATOM, EXTRA_ANGLE_PER_ATOM, EXTRA_DIHEDRAL_PER_ATOM,
    EXTRA_IMPROPER_PER_ATOM, EXTRA_SPECIAL_PER_ATOM, ATOM_MAXSPECIAL,
    ATOM_MAXEXCHANGE, LABELMAP,
})
_BIGINT_FLAGS = frozenset({
    NTIMESTEP, NATOMS, NBONDS, NANGLES, NDIHEDRALS, NIMPROPERS,
    NELLIPSOIDS, NLINES, NTRIS, NBODIES, ATIMESTEP,
})
_DOUBLE_FLAGS = frozenset({TIMESTEP, XY, XZ, YZ, ATIME, ATOM_SORTBIN, COMM_CUTOFF})
# PAIR and the bonded styles are deliberately absent: their string is followed
# by style-specific binary that has to be consumed too, so they are handled
# separately rather than as a plain string flag.
_STRING_FLAGS = frozenset({VERSION, UNITS, ATOM_STYLE})
_INT_VEC_FLAGS = frozenset({PROCGRID, BOUNDARY})
_DOUBLE_VEC_FLAGS = frozenset({BOXLO, BOXHI, SPECIAL_LJ, SPECIAL_COUL,
                               BOUNDMIN, ROTATE_G2R})

_BOUNDARY_CHARS = {0: "p", 1: "f", 2: "s", 3: "m"}


@dataclass
class RestartData:
    """Everything taichimps can use out of a LAMMPS restart file."""

    timestep: int = 0
    dt: float = 0.0
    natoms: int = 0
    ntypes: int = 1
    dimension: int = 3
    units: str = "si"
    atom_style: str = "sphere"
    boxlo: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    boxhi: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tilt: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    boundary: tuple[str, str, str] = ("p", "p", "p")
    # Pair style name and, for the styles taichimps understands, its settings.
    pair_style: str = ""
    pair_settings: dict = field(default_factory=dict)
    # Opaque global state blobs of fixes that store any, keyed by fix style,
    # and the styles that appended per-atom data to each atom's record.
    fix_global: dict = field(default_factory=dict)
    fix_peratom_styles: list = field(default_factory=list)
    # Per-atom arrays, ordered as they appear in the file.
    tag: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    atom_type: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    omega: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rmass: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Leftover doubles per atom, i.e. whatever fixes appended. Kept so the
    # contact history can be recovered when the layout is known.
    fix_extra: list[np.ndarray] = field(default_factory=list)

    @property
    def density(self) -> np.ndarray:
        """Mass density, since AtomSystem.add_particles takes density."""
        vol = (4.0 / 3.0) * np.pi * self.radius**3
        return np.divide(self.rmass, vol, out=np.zeros_like(self.rmass), where=vol > 0)


class _Reader:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.pos = 0

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.raw):
            raise ValueError("restart file ended mid-record")
        out = self.raw[self.pos : self.pos + n]
        self.pos += n
        return out

    def int32(self) -> int:
        return int(struct.unpack("<i", self._take(4))[0])

    def int64(self) -> int:
        return int(struct.unpack("<q", self._take(8))[0])

    def double(self) -> float:
        return float(struct.unpack("<d", self._take(8))[0])

    def int_vec(self) -> list[int]:
        n = self.int32()
        return list(struct.unpack(f"<{n}i", self._take(4 * n)))

    def double_vec(self) -> list[float]:
        n = self.int32()
        return list(struct.unpack(f"<{n}d", self._take(8 * n)))

    def string(self) -> str:
        n = self.int32()
        return self._take(n).decode("utf-8").rstrip("\x00")

    def doubles(self, n: int) -> np.ndarray:
        return np.frombuffer(self._take(8 * n), dtype="<f8").copy()


def _as_int(value: float) -> int:
    """Undo the LAMMPS ubuf union: an int64 carried in a double's bits."""
    return int(struct.unpack("<q", struct.pack("<d", value))[0])


def read_restart(path: str | Path) -> RestartData:
    """
    Read a LAMMPS binary restart file written for `atom_style sphere`.

    Multi-processor restarts (`write_restart file.%`) are not supported; those
    split the atoms across several files.
    """
    raw = Path(path).read_bytes()
    r = _Reader(raw)

    if r._take(len(MAGIC)) != MAGIC:
        raise ValueError(f"{path} is not a LAMMPS restart file")
    endian = r.int32()
    if endian != ENDIAN:
        raise ValueError(
            f"{path} was written on a machine with the opposite byte order"
        )
    revision = r.int32()
    if revision > FORMAT_REVISION:
        raise ValueError(
            f"{path} uses restart format revision {revision}; this reader "
            f"understands up to {FORMAT_REVISION}"
        )

    data = RestartData()
    _read_flag_section(r, data)      # header
    _read_groups(r)                  # Group::write_restart
    _read_flag_section(r, data)      # type arrays (per-type mass)
    _read_flag_section(r, data)      # force field styles
    _read_fix_state(r, data)         # Modify::write_restart
    _read_flag_section(r, data)      # file layout

    _read_atoms(r, data)
    return data


def _read_groups(r: _Reader) -> None:
    """
    Group::write_restart: a count, then that many length-prefixed names.

    Entries for deleted groups are written as a zero length and skipped, which
    is why this counts names rather than iterations.
    """
    ngroup = r.int32()
    found = 0
    while found < ngroup:
        n = r.int32()
        if n:
            r._take(n)
            found += 1


def _read_fix_state(r: _Reader, data: RestartData) -> None:
    """
    Modify::write_restart: fixes with global state, then with per-atom state.

    Each global entry is id, style and an opaque byte blob, all length
    prefixed, so an unfamiliar fix can be stepped over. The per-atom entries
    only declare a maximum size here; the data itself rides along in each
    atom's record.
    """
    for _ in range(r.int32()):
        r._take(r.int32())                       # fix id
        style = r._take(r.int32()).rstrip(b"\x00").decode("utf-8")
        blob = r._take(r.int32())
        data.fix_global[style] = blob
    for _ in range(r.int32()):
        r._take(r.int32())                       # fix id
        style = r._take(r.int32()).rstrip(b"\x00").decode("utf-8")
        r.int32()                                # maxsize_restart
        data.fix_peratom_styles.append(style)


def _read_flag_section(r: _Reader, data: RestartData) -> None:
    """Read (flag, payload) pairs until the -1 terminator."""
    while True:
        flag = r.int32()
        if flag < 0:
            return
        if flag in _STRING_FLAGS:
            value_s = r.string()
            if flag == UNITS:
                data.units = value_s
            elif flag == ATOM_STYLE:
                data.atom_style = value_s.split()[0]
                # The style string is followed by its sub-style arguments,
                # written raw rather than behind a flag: an int count and then
                # that many strings (WriteRestart::header).
                for _ in range(r.int32()):
                    r.string()
        elif flag in _BIGINT_FLAGS:
            value_i = r.int64()
            if flag == NTIMESTEP:
                data.timestep = value_i
            elif flag == NATOMS:
                data.natoms = value_i
        elif flag in _INT_FLAGS:
            value_i = r.int32()
            if flag == DIMENSION:
                data.dimension = value_i
            elif flag == NTYPES:
                data.ntypes = value_i
            elif flag == MULTIPROC and value_i:
                raise ValueError(
                    "multi-processor restart files are not supported; write a "
                    "single-file restart instead"
                )
        elif flag in _DOUBLE_FLAGS:
            value_d = r.double()
            if flag == TIMESTEP:
                data.dt = value_d
            elif flag == XY:
                data.tilt[0] = value_d
            elif flag == XZ:
                data.tilt[1] = value_d
            elif flag == YZ:
                data.tilt[2] = value_d
        elif flag in _INT_VEC_FLAGS:
            vec_i = r.int_vec()
            if flag == BOUNDARY:
                # Six entries, lo/hi per dimension; taichimps has one per dim.
                data.boundary = tuple(  # type: ignore[assignment]
                    _BOUNDARY_CHARS.get(vec_i[2 * d], "f") for d in range(3)
                )
        elif flag in _DOUBLE_VEC_FLAGS:
            vec_d = r.double_vec()
            if flag == BOXLO:
                data.boxlo = vec_d
            elif flag == BOXHI:
                data.boxhi = vec_d
        elif flag == MASS:
            r.double_vec()
        elif flag in (PAIR, NO_PAIR):
            data.pair_style = r.string()
            if flag == PAIR:
                _read_pair_restart(r, data)
        else:
            raise ValueError(
                f"unknown restart header flag {flag}; the file may have been "
                "written by an incompatible LAMMPS version"
            )


# GranularModel writes one record per sub-model slot, in this order.
_GRAN_SUBMODELS = ("normal", "damping", "tangential", "rolling", "twisting", "heat")


def _read_pair_restart(r: _Reader, data: RestartData) -> None:
    """
    Read the pair style's own restart payload, which carries no flags.

    The bytes have to be consumed exactly, or everything after the force field
    section is misread, so only the styles whose layout is known are accepted.
    """
    style = data.pair_style
    if style == "granular":
        models = []
        for _ in range(r.int32()):
            model: dict = {}
            for slot in _GRAN_SUBMODELS:
                name = r._take(r.int32()).decode("utf-8")  # not NUL terminated
                model[slot] = (name, [r.double() for _ in range(r.int32())])
            model["limit_damping"] = r.int32()
            models.append(model)
        # setflag / cutoff / model index per type pair, upper triangle
        ntypes = data.ntypes
        for i in range(ntypes):
            for _j in range(i, ntypes):
                if r.int32():
                    r.double()
                    r.int32()
        data.pair_settings = {"models": models}
    elif style in ("gran/hooke", "gran/hooke/history", "gran/hertz/history"):
        settings = {
            "kn": r.double(), "kt": r.double(),
            "gamman": r.double(), "gammat": r.double(),
            "xmu": r.double(), "dampflag": r.int32(),
            "limit_damping": r.int32(),
        }
        ntypes = data.ntypes
        for i in range(ntypes):
            for _j in range(i, ntypes):
                r.int32()
        data.pair_settings = settings
    else:
        raise ValueError(
            f"restart file stores pair_style {style!r}, whose binary layout "
            "this reader does not know; its bytes cannot be skipped safely"
        )


def _read_atoms(r: _Reader, data: RestartData) -> None:
    """Walk the per-processor chunks and unpack each atom record."""
    tag, atom_type = [], []
    x, v, omega, radius, rmass = [], [], [], [], []
    extra: list[np.ndarray] = []

    # write_restart.cpp closes the file with a second magic string after the
    # last chunk, so the chunk loop has to stop on it rather than on EOF.
    while r.pos + len(MAGIC) <= len(r.raw) and r.raw[r.pos : r.pos + len(MAGIC)] != MAGIC:
        flag = r.int32()
        if flag != PERPROC:
            raise ValueError(f"expected a PERPROC chunk, got flag {flag}")
        n = r.int32()
        buf = r.doubles(n)

        m = 0
        while m < n:
            size = int(buf[m])
            if size < 16:
                raise ValueError(
                    f"atom record of {size} doubles is too short for "
                    "atom_style sphere"
                )
            rec = buf[m : m + size]
            x.append(rec[1:4])
            tag.append(_as_int(rec[4]))
            atom_type.append(_as_int(rec[5]))
            # rec[6] mask, rec[7] image -- taichimps tracks neither
            v.append(rec[8:11])
            radius.append(rec[11])
            rmass.append(rec[12])
            omega.append(rec[13:16])
            extra.append(rec[16:].copy())
            m += size

    data.tag = np.asarray(tag, dtype=np.int32)
    data.atom_type = np.asarray(atom_type, dtype=np.int32)
    data.x = np.asarray(x, dtype=np.float64)
    data.v = np.asarray(v, dtype=np.float64)
    data.omega = np.asarray(omega, dtype=np.float64)
    data.radius = np.asarray(radius, dtype=np.float64)
    data.rmass = np.asarray(rmass, dtype=np.float64)
    data.fix_extra = extra

    if data.natoms and len(data.tag) != data.natoms:
        raise ValueError(
            f"restart header declares {data.natoms} atoms but "
            f"{len(data.tag)} were found"
        )


def parse_neigh_history(record: np.ndarray, dnum: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode one FixNeighHistory per-atom restart record.

    Layout, from FixNeighHistory::pack_restart():
        [m, npartner, tag_0, value_0[dnum], tag_1, value_1[dnum], ...]
    `dnum` is 3 for the shear history of the gran/* pair styles.

    Returns (partner tags, values with shape (npartner, dnum)).
    """
    if record.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, dnum))
    npartner = int(record[1])
    tags = np.zeros(npartner, dtype=np.int32)
    values = np.zeros((npartner, dnum), dtype=np.float64)
    m = 2
    for k in range(npartner):
        tags[k] = _as_int(record[m])
        values[k] = record[m + 1 : m + 1 + dnum]
        m += 1 + dnum
    return tags, values


class _Writer:
    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def raw(self, payload: bytes) -> None:
        self.parts.append(payload)

    def flag_int(self, flag: int, value: int) -> None:
        self.parts.append(struct.pack("<ii", flag, value))

    def flag_bigint(self, flag: int, value: int) -> None:
        self.parts.append(struct.pack("<iq", flag, value))

    def flag_double(self, flag: int, value: float) -> None:
        self.parts.append(struct.pack("<id", flag, value))

    def flag_string(self, flag: int, value: str) -> None:
        payload = value.encode("utf-8") + b"\x00"
        self.parts.append(struct.pack("<ii", flag, len(payload)) + payload)

    def flag_int_vec(self, flag: int, values: list[int]) -> None:
        self.parts.append(
            struct.pack("<ii", flag, len(values))
            + struct.pack(f"<{len(values)}i", *values)
        )

    def flag_double_vec(self, flag: int, values: list[float]) -> None:
        self.parts.append(
            struct.pack("<ii", flag, len(values))
            + struct.pack(f"<{len(values)}d", *values)
        )

    def end_section(self) -> None:
        self.parts.append(struct.pack("<i", -1))

    def getvalue(self) -> bytes:
        return b"".join(self.parts)


def _as_double(value: int) -> float:
    """The LAMMPS ubuf union, the other way: an int64 in a double's bits."""
    return float(struct.unpack("<d", struct.pack("<q", int(value)))[0])


def write_restart(
    path: str | Path,
    atom: object,
    domain: object,
    timestep: int,
    dt: float,
    history: object | None = None,
    units: str = "si",
    pair: object | None = None,
    pair_style: str = "granular",
    nlist: object | None = None,
) -> None:
    """
    Write a single-processor LAMMPS restart file for `atom_style sphere`.

    The contact history is written in the FixNeighHistory layout when a
    ContactHistory is given, so a restart carries frictional state the same way
    LAMMPS does, and the real LAMMPS reads it back into its own history arrays.

    Pass `pair` (a PairGranular) to store the contact model too, the way
    `write_restart` does; without it the file carries no force field section and
    whatever reads it has to set the pair style itself.

    `nlist` is needed alongside `history`: the contact history is stored per
    pair, so which atom a contact belongs to is only known from the neighbour
    list's pair arrays.
    """
    n = atom.nlocal  # type: ignore[attr-defined]
    w = _Writer()
    w.raw(MAGIC)
    w.raw(struct.pack("<ii", ENDIAN, FORMAT_REVISION))

    # --- header
    w.flag_string(VERSION, VERSION_STRING)
    w.flag_int(SMALLINT, 4)
    w.flag_int(IMAGEINT, 4)
    w.flag_int(TAGINT, 4)
    w.flag_int(BIGINT, 8)
    w.flag_string(UNITS, units)
    w.flag_bigint(NTIMESTEP, int(timestep))
    w.flag_int(DIMENSION, 3)
    w.flag_int(NPROCS, 1)
    w.flag_int_vec(PROCGRID, [1, 1, 1])
    # taichimps evaluates every pair once and applies both halves itself, which
    # is what `newton off` means for a granular run; saying so here keeps
    # LAMMPS from warning about a mismatch when it reads the file back.
    w.flag_int(NEWTON_PAIR, 0)
    w.flag_int(NEWTON_BOND, 0)
    periodicity = domain.periodicity  # type: ignore[attr-defined]
    w.flag_int(XPERIODIC, int(periodicity[0]))
    w.flag_int(YPERIODIC, int(periodicity[1]))
    w.flag_int(ZPERIODIC, int(periodicity[2]))
    w.flag_int_vec(
        BOUNDARY, [0 if periodicity[d] else 1 for d in range(3) for _ in range(2)]
    )
    w.flag_string(ATOM_STYLE, "sphere")
    # WriteRestart::header follows the style name with its sub-style arguments:
    # a raw count, then that many strings. atom_style sphere takes none.
    w.raw(struct.pack("<i", 0))
    w.flag_bigint(NATOMS, n)
    ntypes = int(atom.atom_type.to_numpy()[:n].max()) if n else 1  # type: ignore[attr-defined]
    w.flag_int(NTYPES, ntypes)
    w.flag_int(TRICLINIC, 1 if domain.triclinic else 0)  # type: ignore[attr-defined]
    w.flag_double_vec(BOXLO, [float(v) for v in domain.boxlo])  # type: ignore[attr-defined]
    w.flag_double_vec(BOXHI, [float(v) for v in domain.boxhi])  # type: ignore[attr-defined]
    if domain.triclinic:  # type: ignore[attr-defined]
        w.flag_double(XY, float(domain.tilt[0]))  # type: ignore[attr-defined]
        w.flag_double(XZ, float(domain.tilt[1]))  # type: ignore[attr-defined]
        w.flag_double(YZ, float(domain.tilt[2]))  # type: ignore[attr-defined]
    w.flag_double(TIMESTEP, float(dt))
    w.end_section()

    # --- groups: LAMMPS always has "all" as group 0
    w.raw(struct.pack("<i", 1))
    w.raw(struct.pack("<i", 4) + b"all\x00")

    # --- type arrays (none: sphere carries per-atom mass)
    w.end_section()

    # --- force fields
    if pair is not None:
        _write_pair_restart(w, pair, pair_style, ntypes)
    w.end_section()

    # --- fix state: no global blobs; NEIGH_HISTORY rides in the atom records
    w.raw(struct.pack("<i", 0))
    if history is not None:
        fix_id = _HISTORY_FIX_ID.get(pair_style, _HISTORY_FIX_ID["granular"])
        w.raw(struct.pack("<i", 1))
        _write_counted_string(w, fix_id)
        _write_counted_string(w, "NEIGH_HISTORY")
        w.raw(struct.pack(
            "<i",
            _neigh_history_maxsize(
                history, nlist, atom.tag.to_numpy()[:n], n  # type: ignore[attr-defined]
            ),
        ))
    else:
        w.raw(struct.pack("<i", 0))

    # --- file layout
    w.flag_int(MULTIPROC, 0)
    w.end_section()

    # --- one PERPROC chunk
    payload = _pack_atoms(atom, history, nlist, n)
    w.parts.append(struct.pack("<ii", PERPROC, len(payload)))
    w.parts.append(payload.astype("<f8").tobytes())
    w.raw(MAGIC)

    Path(path).write_bytes(w.getvalue())


def _write_counted_string(w: _Writer, value: str) -> None:
    """A NUL-terminated string behind its length, the way LAMMPS stores ids."""
    payload = value.encode("utf-8") + b"\x00"
    w.raw(struct.pack("<i", len(payload)) + payload)


def _write_pair_restart(w: _Writer, pair: object, style: str, ntypes: int) -> None:
    """The force field section, mirroring WriteRestart::force_fields()."""
    submodels = getattr(pair, "submodels", None)
    if style != "granular" or submodels is None:
        raise ValueError(
            f"storing pair_style {style!r} in a restart file is not implemented; "
            "omit `pair` to write the file without a force field section"
        )
    w.flag_string(PAIR, "granular")
    w.raw(struct.pack("<i", 1))                       # one model
    for slot in _GRAN_SUBMODELS:
        name, coeffs = submodels[slot]
        raw = name.encode("utf-8")                    # not NUL terminated here
        w.raw(struct.pack("<i", len(raw)) + raw)
        w.raw(struct.pack("<i", len(coeffs)))
        w.raw(struct.pack(f"<{len(coeffs)}d", *coeffs) if coeffs else b"")
    w.raw(struct.pack("<i", int(getattr(pair, "limit_damping", 0))))
    # setflag / cutoff / model index for every type pair in the upper triangle.
    # -1 is the "no explicit cutoff" value PairGranular::coeff() stores when the
    # pair_coeff line has no `cutoff` keyword, which makes init_one() derive the
    # cutoff from the particle radii; 0 would leave it with no cutoff at all.
    for _i in range(ntypes):
        for _j in range(_i, ntypes):
            w.raw(struct.pack("<idi", 1, -1.0, 0))


def _neigh_history_maxsize(history: object, nlist: object, tag: np.ndarray,
                           n: int) -> int:
    """The largest per-atom FixNeighHistory record, in doubles."""
    contacts = _both_sided_history(history, nlist, tag, n)
    if not contacts:
        return 2
    return 2 + 4 * max(len(c) for c in contacts)


def _pack_atoms(atom: object, history: object | None, nlist: object | None,
                n: int) -> np.ndarray:
    x = atom.x.to_numpy()[:n]        # type: ignore[attr-defined]
    v = atom.v.to_numpy()[:n]        # type: ignore[attr-defined]
    omega = atom.omega.to_numpy()[:n]  # type: ignore[attr-defined]
    radius = atom.radius.to_numpy()[:n]  # type: ignore[attr-defined]
    rmass = atom.rmass.to_numpy()[:n]    # type: ignore[attr-defined]
    tag = atom.tag.to_numpy()[:n]        # type: ignore[attr-defined]
    atype = atom.atom_type.to_numpy()[:n]  # type: ignore[attr-defined]

    contacts = _both_sided_history(history, nlist, tag, n)

    out: list[float] = []
    for i in range(n):
        rec = [
            0.0,
            float(x[i, 0]), float(x[i, 1]), float(x[i, 2]),
            _as_double(tag[i]), _as_double(atype[i]),
            _as_double(1), _as_double(_IMAGE_ZERO),
            float(v[i, 0]), float(v[i, 1]), float(v[i, 2]),
            float(radius[i]), float(rmass[i]),
            float(omega[i, 0]), float(omega[i, 1]), float(omega[i, 2]),
        ]
        if contacts is not None:
            hist = [0.0, float(len(contacts[i]))]
            for ptag, value in contacts[i]:
                hist.append(_as_double(ptag))
                hist.extend(float(c) for c in value)
            hist[0] = float(len(hist))
            rec.extend(hist)
        rec[0] = float(len(rec))
        out.extend(rec)
    return np.asarray(out, dtype=np.float64)


def _both_sided_history(
    history: object, nlist: object, tag: np.ndarray, n: int
) -> list[list[tuple[int, np.ndarray]]] | None:
    """
    Expand taichimps' one entry per contact into the two LAMMPS stores.

    taichimps keeps the neighbor list half, so a contact lives in exactly one
    slot, under whichever of the two atoms owns it.  LAMMPS keeps it under both
    (FixNeighHistory::pre_exchange_no_newton), with the partner's copy negated
    because the shear is measured from i towards j.  Writing only our half
    would load back with every second contact's history silently zeroed, since
    FixNeighHistory::post_neighbor looks the partner up under each atom in turn.
    """
    if history is None or nlist is None:
        return None
    own = history.touching_by_atom(nlist, n)  # type: ignore[attr-defined]

    index_of_tag = {int(t): i for i, t in enumerate(tag)}
    out: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n)]
    for i in range(n):
        for ptag, value in own[i]:
            out[i].append((ptag, value))
            j = index_of_tag.get(ptag)
            if j is not None:
                out[j].append((int(tag[i]), -value))
    return out


# The fix id LAMMPS gives its own FixNeighHistory, by pair style.  The per-atom
# restart data is matched on id *and* style when a file is read back, so a file
# written under the wrong id loads with "Unused restart file peratom fix info"
# and silently loses every contact.
# See PairGranular::init_style and PairGranHookeHistory::set_history_id; the
# trailing 0 is Pair::instance_index() for a single, non-hybrid pair style.
_HISTORY_FIX_ID = {
    "granular": "NEIGH_HISTORY_GRANULAR0",
    "gran/hooke/history": "NEIGH_HISTORY_HH0",
    "gran/hertz/history": "NEIGH_HISTORY_HH0",
}


# LAMMPS packs image flags as (512,512,512) in three 10-bit fields, i.e. the
# "no periodic crossing yet" value.
_IMAGE_ZERO = (512 << 20) | (512 << 10) | 512
