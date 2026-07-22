#!/usr/bin/env python3
"""Bump __version__ in version.py, commit it, and tag the release.

Usage: python3 bump_version.py <major|minor|patch|X.Y.Z>

Does not push -- run `git push && git push origin vX.Y.Z` yourself once
you're ready.
"""
import re
import subprocess
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent / "version.py"
VERSION_RE = re.compile(r'__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)"')


def _current_version():
    match = VERSION_RE.search(VERSION_FILE.read_text())
    if not match:
        sys.exit(f"Could not find __version__ in {VERSION_FILE}")
    return tuple(int(part) for part in match.groups())


def _next_version(current, bump):
    major, minor, patch = current
    if bump == "major":
        return (major + 1, 0, 0)
    if bump == "minor":
        return (major, minor + 1, 0)
    if bump == "patch":
        return (major, minor, patch + 1)
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", bump)
    if not match:
        sys.exit(f"'{bump}' is not 'major', 'minor', 'patch', or an X.Y.Z version")
    return tuple(int(part) for part in match.groups())


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <major|minor|patch|X.Y.Z>")

    new_version = "%d.%d.%d" % _next_version(_current_version(), sys.argv[1])
    VERSION_FILE.write_text(VERSION_RE.sub(f'__version__ = "{new_version}"', VERSION_FILE.read_text()))

    subprocess.run(["git", "add", str(VERSION_FILE)], check=True)
    subprocess.run(["git", "commit", "-m", f"Bump version to {new_version}"], check=True)
    subprocess.run(["git", "tag", "-a", f"v{new_version}", "-m", f"v{new_version}"], check=True)

    print(f"Bumped to {new_version} and tagged v{new_version} locally.")
    print(f"Push when ready: git push && git push origin v{new_version}")


if __name__ == "__main__":
    main()
