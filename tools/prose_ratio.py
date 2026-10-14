"""Share of each package's source lines that are prose: docstrings and comments.

    python tools/prose_ratio.py                 # every package
    python tools/prose_ratio.py neu-lib -v      # one package, per file

A line counts as prose if it is part of a docstring (module, class or function, found by
parsing) or starts with ``#``. The ratio is prose lines over non-blank lines, with tests
excluded.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["neu-lib", "blockrun", "neu-vol", "neu-morpho", "neu-mark", "neu-glance",
            "neu-draw", "neu-proc", "neu-eval"]


def count(path: Path) -> tuple[int, int]:
    """(prose lines, non-blank lines) for one file."""
    text = path.read_text()
    lines = text.splitlines()
    prose: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                prose.update(range(body[0].lineno, body[0].end_lineno + 1))
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            prose.add(i)
    nonblank = {i for i, line in enumerate(lines, 1) if line.strip()}
    return len(prose & nonblank), len(nonblank)


def package_files(pkg: str) -> list[Path]:
    src = ROOT / pkg / pkg.replace("-", "_")
    return sorted(p for p in src.rglob("*.py") if "tests" not in p.parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("packages", nargs="*", default=PACKAGES)
    parser.add_argument("-v", "--verbose", action="store_true", help="per-file breakdown")
    args = parser.parse_args(argv)
    for pkg in args.packages:
        total_prose = total = 0
        for path in package_files(pkg):
            p, n = count(path)
            total_prose += p
            total += n
            if args.verbose:
                print(f"  {p / n:5.0%}  {p:5d}/{n:<5d}  {path.relative_to(ROOT)}")
        print(f"{pkg:<11} {total_prose / total:5.0%}  ({total_prose} of {total} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
