from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ms_swift_grpo_passes_automatically_discovered_checkpoint(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "args.txt"
    checkpoint = tmp_path / "output" / "v0-20260902-010000" / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    fake_swift = fake_bin / "swift"
    fake_swift.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    fake_swift.chmod(0o755)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-m" && "${2:-}" == '
        '"graphtask_r1.training.checkpointing" ]]; then\n'
        '  printf "%s\\n" "$FAKE_CHECKPOINT"\n'
        "fi\n"
    )
    fake_python.chmod(0o755)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    train_data = tmp_path / "train.parquet"
    train_data.touch()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_ARGS": str(capture),
        "FAKE_CHECKPOINT": str(checkpoint),
        "LORA_ADAPTER_PATH": str(adapter),
        "TRAIN_DATA": str(train_data),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "USE_VLLM": "false",
    }

    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/train_ms_swift_grpo.sh")],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = capture.read_text().splitlines()
    resume_index = arguments.index("--resume_from_checkpoint")
    assert arguments[resume_index + 1] == str(checkpoint)
