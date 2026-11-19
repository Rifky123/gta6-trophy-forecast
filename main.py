"""gta6-trophy-rarity-forecast-01 entry: earners-over-players forecaster with bootstrap spread.

Reads every <id>.in in the input directory (the grader runs one case at a time, with only that
case's file there) and writes <id>.out beside it: for each of the case's trophies, five quantiles
of its earn rate at the case's target, in percent. See model.py for the forecast itself.

Runs as `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim, standard library
only: no network, no pip packages. Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds model.py
# shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
