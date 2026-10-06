"""Fail if any tracked source file exceeds 1,000 lines."""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES = 1000
ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "tests", "scripts", "docs", "notebooks")
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".parquet", ".csv", ".zip", ".ckpt"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for name in SCAN_DIRS:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix not in SKIP_SUFFIXES:
                files.append(path)
    for name in ("README.md", "AGENT.md", "handover.md"):
        path = ROOT / name
        if path.exists():
            files.append(path)
    files.extend(ROOT.glob("WEEK_*.md"))
    return files


def main() -> int:
    offenders: list[tuple[Path, int]] = []
    for path in iter_files():
        line_count = sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))
        if line_count > MAX_LINES:
            offenders.append((path.relative_to(ROOT), line_count))

    if offenders:
        print(f"Files exceeding {MAX_LINES} lines:")
        for path, count in sorted(offenders, key=lambda item: item[1], reverse=True):
            print(f"  {path}: {count} lines")
        return 1

    print(f"All scanned files are <= {MAX_LINES} lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
