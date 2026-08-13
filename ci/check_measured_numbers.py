"""CI check: post drafts must not contain unsourced numbers (spec §8, kickoff agreement 3:
"measured numbers only").

Scope: ``docs/drafts/**/*.md`` — the post-draft area. Working docs (TASKS/DEVLOG/NOTES)
legitimately contain dates, task IDs, and PR numbers and are out of scope.

A line containing a digit is flagged unless it carries one of:
- ``measured:`` — a source tag pointing at a run/log/bill,
- ``MEASURED_TBD`` — the sanctioned placeholder,
- ``<!-- no-number -->`` — explicit per-line allowlist marker for non-claim numbers
  (e.g. "Go1", "3D", section references).

This heuristic is deliberately strict and will be tuned when the first real draft lands
(tuning goes through DEVLOG with a DECISION line).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DRAFTS_DIR = Path("docs/drafts")
ALLOW_MARKERS = ("measured:", "MEASURED_TBD", "<!-- no-number -->")
DIGIT_RE = re.compile(r"\d")


def flag_line(line: str) -> bool:
    """True if the line contains a number without a sanctioned source/placeholder."""
    if not DIGIT_RE.search(line):
        return False
    return not any(marker in line for marker in ALLOW_MARKERS)


def check(repo_root: Path) -> tuple[bool, list[str]]:
    """Return (ok, violations). Each violation is 'path:lineno: line'."""
    drafts = repo_root / DRAFTS_DIR
    violations: list[str] = []
    if not drafts.exists():
        return True, violations
    for path in sorted(drafts.rglob("*.md")):
        relative = path.relative_to(repo_root)
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if flag_line(line):
                violations.append(f"{relative}:{lineno}: {line.strip()}")
    return len(violations) == 0, violations


def main() -> int:
    ok, violations = check(Path.cwd())
    if ok:
        print("measured-numbers: no unsourced numbers in docs/drafts — ok")
        return 0
    print("measured-numbers: unsourced numbers found in post drafts (spec §8):")
    for violation in violations:
        print(f"  {violation}")
    print("Tag with 'measured: <source>', use MEASURED_TBD, or mark '<!-- no-number -->'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
