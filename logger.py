"""
BugAgent structured logging.

Two-tier logging:
  - File  (DEBUG)  → data/logs/<video>_<timestamp>.log  — full detail
  - Console (INFO) → stdout — stage progress only

Usage in modules:
    from logger import get_logger
    _log = get_logger(__name__)
    _log.info("...")
    _log.debug("...")        # file only
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Formatters ────────────────────────────────────────────────────────────────

_ANSI = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class _ConsoleFormatter(logging.Formatter):
    """Compact, optionally coloured console formatter."""

    def format(self, record: logging.LogRecord) -> str:
        lvl = record.levelname
        if sys.stdout.isatty():
            color = _ANSI.get(lvl, "")
            record.levelname = f"{color}{lvl:<7}{_ANSI['RESET']}"
        else:
            record.levelname = f"{lvl:<7}"
        return super().format(record)


_FILE_FMT = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_CONSOLE_FMT = _ConsoleFormatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ── Public API ────────────────────────────────────────────────────────────────

def setup_logging(
    log_dir: str = "data/logs",
    video_name: str = "run",
    verbose: bool = True,
) -> str:
    """
    Configure logging for a BugAgent run.

    Args:
        log_dir:    Directory to write the log file into.
        video_name: Used to name the log file.
        verbose:    If True, console shows INFO+; otherwise WARNING+.

    Returns:
        Absolute path to the created log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{video_name}_{ts}.log"

    root = logging.getLogger("bugagent")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()          # avoid duplicate handlers on re-init

    # ── File handler: captures everything ──
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FILE_FMT)
    root.addHandler(fh)

    # ── Console handler: progress only ──
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(_CONSOLE_FMT)
    root.addHandler(ch)

    root.info(f"Log file: {log_file.resolve()}")
    return str(log_file.resolve())


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'bugagent' namespace.

    Args:
        name: Usually __name__ of the calling module.
    """
    # Strip leading package path to keep names short in file
    short = name.replace("bugagent.", "")
    return logging.getLogger(f"bugagent.{short}")
