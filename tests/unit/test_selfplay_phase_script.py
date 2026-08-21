from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/run_selfplay_curriculum_phases.sh"


def test_selfplay_phase_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_selfplay_phase_script_continues_after_a_failed_command(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "commands.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$CAPTURE_ARGS"\n'
        'if [[ "$*" == *"--round-index 1 --phase questioner"* ]]; then\n'
        "  exit 17\n"
        "fi\n"
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_ARGS": str(capture),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT), "config.yaml", str(tmp_path / "output")],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    commands = capture.read_text().splitlines()
    assert result.returncode == 1
    assert len(commands) == 6
    assert "--round-index 1 --phase solver" in commands[1]
    assert "--round-index 3 --phase solver" in commands[-1]
    assert "continuing to the next command" in result.stderr
    assert "all six commands were attempted; 1 returned" in result.stderr
