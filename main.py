"""Run from any directory: python3 /path/to/every_example_gui/main.py."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    try:
        from every_example_gui.app import run
    except ImportError as exc:
        print(
            f"Could not load the desktop interface: {exc}\nUse Python 3.10+ with Tk installed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    run()


if __name__ == "__main__":
    main()
