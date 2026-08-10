import json
import sys
from pathlib import Path

from graphtask_r1.training.merge_sft import export_command, patch_lora_alpha


def test_export_command_uses_current_python_and_fsdp(tmp_path: Path) -> None:
    command = export_command(tmp_path / "checkpoint", tmp_path / "export")
    assert command[:5] == [sys.executable, "-m", "verl.model_merger", "merge", "--backend"]
    assert command[5] == "fsdp"
    assert "--use_cpu_initialization" in command


def test_patch_lora_alpha_preserves_adapter_config(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config_path = adapter / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 32, "lora_alpha": 0, "target_modules": ["q_proj"]}))
    result = patch_lora_alpha(adapter, 64)
    saved = json.loads(config_path.read_text())
    assert result["lora_alpha"] == 64
    assert saved == {"r": 32, "lora_alpha": 64, "target_modules": ["q_proj"]}
