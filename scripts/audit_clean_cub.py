"""Audit every CUB image and publish separate logical-dedup indexes. No training."""
import argparse
import json
from pathlib import Path

from pattern_recognition_coursework.data.cub_dedup import audit_and_clean

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'cub2002011/versions/7/CUB_200_2011')
    parser.add_argument('--source-splits', type=Path, default=ROOT/'data/splits')
    parser.add_argument('--clean-splits', type=Path, default=ROOT/'data/splits/cub_clean_v1')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'outputs/stage01_data/cub_dedup')
    args = parser.parse_args(argv)
    for path in (args.clean_splits,args.report_dir):
        if not path.resolve().is_relative_to(ROOT):
            parser.error('Derived indexes and reports must remain inside the project')
    result = audit_and_clean(args.data_root,args.source_splits,args.clean_splits,args.report_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    main()
