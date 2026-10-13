"""Lockstep versions for the nine packages.

    python tools/release.py --check     # CI: every version literal agrees
    python tools/release.py 0.2.0       # bump all of them, commit, tag v0.2.0

Each package carries its version twice, as ``version`` in ``pyproject.toml`` and as
``__version__`` in its ``__init__.py``; both are literals, so ``pip install
git+...#subdirectory=<pkg>`` needs no build hook. A release rewrites all eighteen, moves
``CHANGELOG.md``'s ``## Unreleased`` heading under the new version if there is one, then
commits and tags. It never pushes.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["neu-lib", "blockrun", "neu-vol", "neu-morpho", "neu-mark", "neu-glance",
            "neu-draw", "neu-proc", "neu-eval"]
PYPROJECT_RE = re.compile(r'^version = "([^"]+)"$', re.M)
INIT_RE = re.compile(r'^__version__ = "([^"]+)"$', re.M)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def version_files() -> list[tuple[Path, re.Pattern]]:
    files = []
    for pkg in PACKAGES:
        files.append((ROOT / pkg / "pyproject.toml", PYPROJECT_RE))
        files.append((ROOT / pkg / pkg.replace("-", "_") / "__init__.py", INIT_RE))
    return files


def current_versions() -> dict[str, str]:
    """Relative path -> version literal, for every file that carries one."""
    found = {}
    for path, pattern in version_files():
        matches = pattern.findall(path.read_text())
        if len(matches) != 1:
            sys.exit(f"{path.relative_to(ROOT)}: expected one version literal, "
                     f"found {len(matches)}")
        found[str(path.relative_to(ROOT))] = matches[0]
    return found


def check() -> int:
    found = current_versions()
    distinct = sorted(set(found.values()))
    if len(distinct) == 1:
        print(f"all {len(found)} version literals are {distinct[0]}")
        return 0
    for rel, version in sorted(found.items()):
        print(f"  {version:>10}  {rel}")
    print(f"version literals disagree: {', '.join(distinct)}", file=sys.stderr)
    return 1


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout


def release(version: str) -> int:
    if not SEMVER_RE.match(version):
        sys.exit(f"{version!r} is not X.Y.Z")
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("the working tree has uncommitted changes; commit or stash them first")
    if _git("tag", "--list", f"v{version}").strip():
        sys.exit(f"tag v{version} already exists")
    if check() != 0:
        sys.exit("fix the disagreement above before releasing")

    changed = []
    for path, pattern in version_files():
        text = path.read_text()
        prefix = "version = " if pattern is PYPROJECT_RE else "__version__ = "
        path.write_text(pattern.sub(f'{prefix}"{version}"', text))
        changed.append(path)
    changelog = ROOT / "CHANGELOG.md"
    if changelog.exists() and "## Unreleased" in changelog.read_text():
        today = datetime.date.today().isoformat()
        changelog.write_text(changelog.read_text().replace(
            "## Unreleased", f"## Unreleased\n\n## {version} ({today})", 1))
        changed.append(changelog)

    _git("add", "--", *[str(p.relative_to(ROOT)) for p in changed])
    _git("commit", "-q", "-m", f"Release {version}")
    _git("tag", "-a", f"v{version}", "-m", f"Release {version}")
    print(f"committed and tagged v{version}; push with: git push origin main v{version}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", nargs="?", help="the new version, X.Y.Z")
    parser.add_argument("--check", action="store_true",
                        help="only verify that every version literal agrees")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    if not args.version:
        parser.error("give a version, or --check")
    return release(args.version)


if __name__ == "__main__":
    sys.exit(main())
