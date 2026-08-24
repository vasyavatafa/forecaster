"""Progress bars (tqdm) + colored stage logging.

Used everywhere a "stage" happens: one walk-forward run for a single
(method, column) pair, one step inside that walk-forward loop, one
config inside a grid search. The convention is always the same:
  * a tqdm bar shows which stage is currently in flight
  * a stage that finishes cleanly prints one green "[OK]" line
  * a stage that raises is caught, prints one red "[FAIL]" line with
    the exception, and execution moves on to the next stage instead of
    crashing the whole run
  * a single walk-forward step that raises prints one (rate-limited)
    yellow "[WARN]" line, is recorded as a NaN forecast, and the loop
    continues to the next step
"""
from __future__ import annotations

import sys

from tqdm import tqdm

_USE_COLOR = sys.stdout.isatty()

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"


def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{code}{text}{_RESET}"


def log_ok(stage: str, detail: str = "") -> None:
    msg = _c("[OK]", _GREEN + _BOLD) + f" {stage}"
    if detail:
        msg += f" -- {detail}"
    tqdm.write(msg)


def log_fail(stage: str, error: BaseException | str) -> None:
    msg = _c("[FAIL]", _RED + _BOLD) + f" {stage}: {error}"
    tqdm.write(msg)


def log_warn(stage: str, error: BaseException | str) -> None:
    msg = _c("[WARN]", _YELLOW + _BOLD) + f" {stage}: {error}"
    tqdm.write(msg)


def log_info(msg: str) -> None:
    tqdm.write(_c("[INFO]", _CYAN + _BOLD) + f" {msg}")


def progress(iterable, desc: str, total: int | None = None, leave: bool = True, disable: bool = False):
    return tqdm(iterable, desc=desc, total=total, leave=leave, disable=disable)


class RateLimitedWarner:
    """Prints at most `max_prints` [WARN] lines for a repeating failure,
    then a single summary line for the rest -- keeps a systematically
    failing config from flooding stdout with one line per step."""

    def __init__(self, stage: str, max_prints: int = 3):
        self.stage = stage
        self.max_prints = max_prints
        self.count = 0

    def warn(self, detail: str, error: BaseException | str) -> None:
        self.count += 1
        if self.count <= self.max_prints:
            log_warn(f"{self.stage} | {detail}", error)
        elif self.count == self.max_prints + 1:
            log_warn(self.stage, f"further step failures suppressed (>{self.max_prints})")

    def summary(self) -> str | None:
        if self.count == 0:
            return None
        return f"{self.count} step(s) failed and were recorded as NaN"
