import os
import sys
import runpy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = ROOT_DIR / "Video_Analysis"
VIDEO_MAIN = VIDEO_DIR / "main.py"


def main():
    if not VIDEO_DIR.is_dir():
        print(f"Error: missing Video_Analysis folder at {VIDEO_DIR}", file=sys.stderr)
        return 1

    if not VIDEO_MAIN.is_file():
        print(f"Error: missing {VIDEO_MAIN}", file=sys.stderr)
        return 1

    os.chdir(VIDEO_DIR)
    if str(VIDEO_DIR) not in sys.path:
        sys.path.insert(0, str(VIDEO_DIR))
    print(f"Running {VIDEO_MAIN}...", flush=True)
    sys.argv = [str(VIDEO_MAIN)] + sys.argv[1:]

    runpy.run_path(str(VIDEO_MAIN), run_name="__main__")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
