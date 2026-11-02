"""Safely import an official NABirds archive into data/raw/nabirds."""

import argparse
import json
from pathlib import Path

from pattern_recognition_coursework.data.import_nabirds import import_archive


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Official archive obtained after Cornell's download form")
    args = parser.parse_args(argv)
    try:
        result = import_archive(args.archive, PROJECT_ROOT)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
