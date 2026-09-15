"""
FixPrint implementation for periodic variable evaluation and file output.
Reference: LAMMPS src/fix_print.cpp
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from taichimps.atom import AtomSystem
from taichimps.domain import Domain
from taichimps.EXTRA_FIX.base import Fix


class FixPrint(Fix):
    """Periodically evaluates a formatted string and writes to file or console."""

    def __init__(
        self,
        domain: Domain | None,
        nevery: int,
        template_str: str,
        eval_fn: Callable[[str], str],
        filepath: str | Path | None = None,
        title: str | None = None,
        screen: bool = False,
        fix_id: str = "",
    ) -> None:
        super().__init__(domain)
        self.nevery = max(1, int(nevery))
        self.template_str = template_str
        self.eval_fn = eval_fn
        self.filepath = Path(filepath) if filepath else None
        self.title = title
        self.screen = screen
        self.file_handle: Any = None
        self.step_count: int = 0

        if self.filepath:
            self.file_handle = open(self.filepath, "w")  # noqa: SIM115
            if self.title is not None and self.title.strip() != "":
                clean_title = self.title.strip("\"'")
                self.file_handle.write(f"{clean_title}\n")
            elif self.title is None:
                suffix = f" for fix {fix_id}" if fix_id else ""
                self.file_handle.write(f"# Fix print output{suffix}\n")
            self.file_handle.flush()

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        """
        Evaluate the string and write it out on every Nth timestep.

        The test is on the absolute timestep, as in FixPrint::end_of_step,
        not on a count of invocations.  Counting invocations put the output on
        run-relative steps, so a run resumed from a restart at step 1000000
        printed at 1002000, 1004000 ... where LAMMPS prints at 1010000,
        1020000 ... and the two could not be compared at all.
        """
        self.step_count += 1
        if self.timestep % self.nevery == 0:
            output_str = self.eval_fn(self.template_str)
            if self.file_handle:
                self.file_handle.write(f"{output_str}\n")
                self.file_handle.flush()
            if self.screen:
                print(output_str)

    def __del__(self) -> None:
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()
