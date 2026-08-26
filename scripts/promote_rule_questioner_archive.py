#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphtask_r1.experiments.rule_questioner import promote_rule_questioner_candidates
from graphtask_r1.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote independent fixed-program Questioner candidates."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-difficulty", type=float, default=0.25)
    parser.add_argument("--max-difficulty", type=float, default=0.75)
    parser.add_argument("--min-novelty", type=float, default=0.0)
    args = parser.parse_args()
    summary = promote_rule_questioner_candidates(
        args.candidates,
        args.archive,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        min_novelty=args.min_novelty,
    )
    write_json(args.report, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
