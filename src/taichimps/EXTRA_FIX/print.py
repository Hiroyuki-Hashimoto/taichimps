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
                self.file_handle.write("# Fix print output\n")
            self.file_handle.flush()

    def end_of_step(self, atom: AtomSystem, dt: float) -> None:
        """Evaluate string and write to file at specified intervals."""
        self.step_count += 1
        if self.step_count % self.nevery == 0:
            output_str = self.eval_fn(self.template_str)
            if self.file_handle:
                self.file_handle.write(f"{output_str}\n")
                self.file_handle.flush()
            if self.screen:
                print(output_str)

    def __del__(self) -> None:
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()
