from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graphtask_r1.utils import write_json

_STEP_METRIC_NAMES = frozenset(
    {
        "loss",
        "grad_norm",
        "learning_rate",
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "kl",
    }
)
_STEP_METRIC_PREFIXES = (
    "completions/",
    "rewards/",
    "clip_ratio/",
    "entropy/",
    "rollout_correction/",
)


def _is_step_metric_row(row: Mapping[str, object]) -> bool:
    """Distinguish optimizer/eval events from terminal trainer summaries.

    Once a row is known to be a step event, all of its finite numeric fields are
    retained. This keeps the report compatible with new ms-swift metrics without
    inventing extra steps from the final ``train_runtime`` summary.
    """
    for name in row:
        unprefixed = name.removeprefix("eval_")
        if unprefixed in _STEP_METRIC_NAMES or unprefixed.startswith(
            _STEP_METRIC_PREFIXES
        ):
            return True
    return False


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(value)
    return rows


def find_trainer_log(round_dir: Path, adapter: Path | None = None) -> Path | None:
    """Find the logging.jsonl belonging to the selected checkpoint, not a stale retry."""
    if adapter is not None:
        resolved_round = round_dir.resolve()
        for parent in (adapter.resolve(), *adapter.resolve().parents):
            candidate = parent / "logging.jsonl"
            if candidate.is_file():
                return candidate
            if parent == resolved_round:
                break
    candidates = list(round_dir.rglob("logging.jsonl"))
    return (
        max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        if candidates
        else None
    )


def _trainer_history(path: Path | None) -> list[dict[str, float]]:
    if path is None:
        return []
    history: list[dict[str, float]] = []
    inferred_step = 0
    for row in _read_jsonl(path):
        if not _is_step_metric_row(row):
            continue
        raw_step = _number(row.get("step"))
        if raw_step is None:
            raw_global_step = row.get("global_step/max_steps")
            if isinstance(raw_global_step, str) and "/" in raw_global_step:
                try:
                    raw_step = float(raw_global_step.split("/", maxsplit=1)[0])
                except ValueError:
                    raw_step = None
        if raw_step is None:
            inferred_step += 1
            raw_step = float(inferred_step)
        else:
            inferred_step = max(inferred_step, int(raw_step))
        metrics = {
            name: number
            for name, value in row.items()
            if name != "step" and (number := _number(value)) is not None
        }
        if metrics:
            history.append({"step": raw_step, **metrics})
    return history


def _combined_trainer_history(
    trainer_logs: Mapping[str, Path],
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    step_offset = 0.0
    for phase_index, (_, path) in enumerate(trainer_logs.items()):
        phase_history = _trainer_history(path)
        phase_step_max = 0.0
        for row in phase_history:
            phase_step = float(row["step"])
            phase_step_max = max(phase_step_max, phase_step)
            history.append(
                {
                    **row,
                    "step": step_offset + phase_step,
                    "phase_index": float(phase_index),
                }
            )
        step_offset += phase_step_max
    return history


def _metric_statistics(history: Iterable[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in history:
        for name, value in row.items():
            if name != "step":
                values[name].append(value)
    return {
        name: {
            "mean": sum(series) / len(series),
            "last": series[-1],
            "min": min(series),
            "max": max(series),
        }
        for name, series in sorted(values.items())
    }


def _reward_role_metrics(metrics_dir: Path | None) -> dict[str, dict[str, Any]]:
    weighted_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples: dict[str, int] = defaultdict(int)
    if metrics_dir is None:
        return {}
    for path in sorted(metrics_dir.glob("reward_components.rank-*.jsonl")):
        for event in _read_jsonl(path):
            roles = event.get("roles")
            if not isinstance(roles, dict):
                continue
            for role, raw_role in roles.items():
                if not isinstance(role, str) or not isinstance(raw_role, dict):
                    continue
                raw_samples = raw_role.get("samples")
                role_samples = int(raw_samples) if isinstance(raw_samples, int) else 0
                means = raw_role.get("means")
                if role_samples <= 0 or not isinstance(means, dict):
                    continue
                samples[role] += role_samples
                for name, raw_value in means.items():
                    value = _number(raw_value)
                    if isinstance(name, str) and value is not None:
                        weighted_sums[role][name] += value * role_samples
                        counts[role][name] += role_samples
    return {
        role: {
            "samples": samples[role],
            "means": {
                name: weighted_sums[role][name] / count
                for name, count in sorted(counts[role].items())
                if count
            },
        }
        for role in sorted(samples)
    }


def summarize_selfplay_round(
    round_index: int,
    counts: Mapping[str, int],
    *,
    trainer_log: Path | None,
    reward_metrics_dir: Path | None,
    archive_size_before: int | None = None,
    archive_size_after: int | None = None,
    trainer_logs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    history = (
        _combined_trainer_history(trainer_logs)
        if trainer_logs is not None
        else _trainer_history(trainer_log)
    )
    roles = _reward_role_metrics(reward_metrics_dir)
    questioner = roles.get("questioner", {}).get("means", {})
    solver = roles.get("solver", {}).get("means", {})
    questioner_score = _number(
        questioner.get("unweighted_score") if isinstance(questioner, dict) else None
    )
    solver_score = _number(
        solver.get("unweighted_score") if isinstance(solver, dict) else None
    )
    cooperation_bottleneck = (
        min(questioner_score, solver_score)
        if questioner_score is not None and solver_score is not None
        else None
    )
    return {
        "round": round_index,
        "dataset_counts": dict(counts),
        "archive": {
            "size_before": archive_size_before,
            "size_after": archive_size_after,
            "added": (
                archive_size_after - archive_size_before
                if archive_size_before is not None and archive_size_after is not None
                else None
            ),
        },
        "trainer_log": str(trainer_log) if trainer_log is not None else None,
        "trainer_logs": (
            {phase: str(path) for phase, path in trainer_logs.items()}
            if trainer_logs is not None
            else None
        ),
        "reward_metrics_dir": str(reward_metrics_dir) if reward_metrics_dir is not None else None,
        "training_history": history,
        "trainer_metrics": _metric_statistics(history),
        "roles": roles,
        "cooperation_bottleneck": cooperation_bottleneck,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _role_series(
    rounds: list[dict[str, Any]], role: str, metric: str
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rounds:
        roles = row.get("roles")
        raw_role = roles.get(role) if isinstance(roles, dict) else None
        means = raw_role.get("means") if isinstance(raw_role, dict) else None
        value = _number(means.get(metric)) if isinstance(means, dict) else None
        round_index = _number(row.get("round"))
        if value is not None and round_index is not None:
            x_values.append(round_index)
            y_values.append(value)
    return x_values, y_values


def plot_selfplay_report(report: Mapping[str, Any], output_path: Path) -> None:
    cache_dir = output_path.parent / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib  # type: ignore[import-untyped]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    raw_rounds = report.get("rounds")
    rounds = (
        [row for row in raw_rounds if isinstance(row, dict)]
        if isinstance(raw_rounds, list)
        else []
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    cooperation_ax = axes[0][0]
    for role, label in (("questioner", "Questioner score"), ("solver", "Solver score")):
        x_values, y_values = _role_series(rounds, role, "unweighted_score")
        if x_values:
            cooperation_ax.plot(x_values, y_values, marker="o", label=label)
    bottleneck_x: list[float] = []
    bottleneck_y: list[float] = []
    for row in rounds:
        x_value = _number(row.get("round"))
        y_value = _number(row.get("cooperation_bottleneck"))
        if x_value is not None and y_value is not None:
            bottleneck_x.append(x_value)
            bottleneck_y.append(y_value)
    if bottleneck_x:
        cooperation_ax.plot(
            bottleneck_x,
            bottleneck_y,
            marker="s",
            linestyle="--",
            label="Cooperation bottleneck",
        )
    cooperation_ax.set(title="Role co-improvement by round", xlabel="Round", ylabel="Reward")

    quality_ax = axes[0][1]
    for role, metric, label in (
        ("questioner", "validity", "Questioner validity"),
        ("questioner", "frontier", "Questioner frontier"),
        ("questioner", "novelty", "Questioner novelty"),
        ("questioner", "target_alignment", "Questioner target alignment"),
        ("questioner", "opponent_success_rate", "Frozen Solver success"),
        ("solver", "f1", "Solver F1"),
        ("solver", "exact_match", "Solver exact match"),
    ):
        x_values, y_values = _role_series(rounds, role, metric)
        if x_values:
            quality_ax.plot(x_values, y_values, marker="o", label=label)
    quality_ax.set(title="Task and solving quality", xlabel="Round", ylabel="Mean")

    optimization_ax = axes[1][0]
    steps = report.get("training_history")
    history = [row for row in steps if isinstance(row, dict)] if isinstance(steps, list) else []
    for metric, label in (("loss", "Loss"), ("kl", "KL"), ("grad_norm", "Gradient norm")):
        x_values = []
        y_values = []
        for row in history:
            x_value = _number(row.get("global_step"))
            y_value = _number(row.get(metric))
            if x_value is not None and y_value is not None:
                x_values.append(x_value)
                y_values.append(y_value)
        if x_values:
            optimization_ax.plot(x_values, y_values, label=label)
    optimization_ax.set(title="Optimization dynamics", xlabel="Cumulative step", ylabel="Value")

    rollout_ax = axes[1][1]
    for metric, label in (
        ("reward", "Train reward"),
        ("eval_reward", "Eval reward"),
        ("completions/clipped_ratio", "Completion clipped ratio"),
    ):
        x_values = []
        y_values = []
        for row in history:
            x_value = _number(row.get("global_step"))
            y_value = _number(row.get(metric))
            if x_value is not None and y_value is not None:
                x_values.append(x_value)
                y_values.append(y_value)
        if x_values:
            rollout_ax.plot(x_values, y_values, label=label)
    rollout_ax.set(title="Rollout signal", xlabel="Cumulative step", ylabel="Value")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_selfplay_report(output_dir: Path) -> dict[str, Any]:
    summaries = [
        json.loads(path.read_text())
        for path in sorted(output_dir.glob("round_*/logs/metrics_summary.json"))
    ]
    rounds = sorted(
        (row for row in summaries if isinstance(row, dict)), key=lambda row: int(row["round"])
    )
    history: list[dict[str, Any]] = []
    step_offset = 0
    for summary in rounds:
        round_step_max = 0
        raw_history = summary.get("training_history")
        rows = raw_history if isinstance(raw_history, list) else []
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            round_step = int(float(row.pop("step", 0)))
            round_step_max = max(round_step_max, round_step)
            history.append(
                {
                    "round": int(summary["round"]),
                    "round_step": round_step,
                    "global_step": step_offset + round_step,
                    **row,
                }
            )
        step_offset += round_step_max
    report = {
        "schema_version": 1,
        "cooperation_bottleneck_definition": (
            "minimum of the unweighted Questioner and Solver mean scores; it rises only "
            "when the weaker role improves"
        ),
        "rounds": rounds,
        "training_history": history,
    }
    logs_dir = output_dir / "logs"
    write_json(logs_dir / "selfplay_metrics.json", report)
    _write_jsonl(logs_dir / "round_metrics.jsonl", rounds)
    _write_jsonl(logs_dir / "training_history.jsonl", history)
    plot_path = logs_dir / "selfplay_curves.png"
    plot_selfplay_report(report, plot_path)
    return {
        "metrics": str(logs_dir / "selfplay_metrics.json"),
        "round_metrics": str(logs_dir / "round_metrics.jsonl"),
        "training_history": str(logs_dir / "training_history.jsonl"),
        "plot": str(plot_path),
    }
