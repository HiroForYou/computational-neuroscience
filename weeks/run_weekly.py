"""Run any one of the 18 weekly scripts or list the available exercises."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def weekly_scripts() -> dict[int, Path]:
    weeks_dir = Path(__file__).resolve().parent
    scripts = {}
    for path in weeks_dir.glob("week_*/week_*.py"):
        if path.stem.removeprefix("week_").split("_", 1)[0].isdigit():
            week = int(path.stem.removeprefix("week_").split("_", 1)[0])
            scripts[week] = path
    return scripts


def main() -> None:
    scripts = weekly_scripts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("week", nargs="?", type=int, help="week number (1-18)")
    parser.add_argument("--list", action="store_true", help="list weekly scripts")
    args = parser.parse_args()

    if args.list:
        for week, path in sorted(scripts.items()):
            print(f"{week:02d}: {path.name}")
        return
    if args.week not in scripts:
        first, last = min(scripts), max(scripts)
        parser.error(f"choose a week from {first} to {last}; use --list to see the scripts")

    runpy.run_path(str(scripts[args.week]), run_name="__main__")


if __name__ == "__main__":
    main()
