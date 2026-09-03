#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphtask_r1.training.evidence_selfplay_runner import (
    load_evidence_rl_train_config,
    run_evidence_selfplay_update,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one no-SFT Evidence Self-Play update with ms-swift 3.10.3."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_evidence_rl_train_config(args.config)
    result = run_evidence_selfplay_update(config, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
