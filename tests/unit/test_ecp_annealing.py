from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_ecp_annealed_curriculum import build_annealed_curriculum


def test_annealed_curriculum_partitions_tasks_and_reward_variants(tmp_path: Path) -> None:
    dense_path = tmp_path / "dense.parquet"
    strict_path = tmp_path / "strict.parquet"
    dense_rows = [{"uid": f"task-{index}", "reward_variant": "ecp_v5"} for index in range(6)]
    strict_rows = [
        {"uid": f"task-{index}", "reward_variant": "ecp_v4"} for index in reversed(range(6))
    ]
    pq.write_table(pa.Table.from_pylist(dense_rows), dense_path)
    pq.write_table(pa.Table.from_pylist(strict_rows), strict_path)

    manifest = build_annealed_curriculum(
        dense_path,
        strict_path,
        tmp_path / "curriculum",
        dense_size=2,
        seed=71,
    )
    stage_one = pq.read_table(
        tmp_path / "curriculum" / "stage_1_dense" / "solver_train.parquet"
    ).to_pylist()
    stage_two = pq.read_table(
        tmp_path / "curriculum" / "stage_2_strict" / "solver_train.parquet"
    ).to_pylist()

    assert len(stage_one) == 2
    assert len(stage_two) == 4
    assert {row["uid"] for row in stage_one}.isdisjoint(row["uid"] for row in stage_two)
    assert {row["reward_variant"] for row in stage_one} == {"ecp_v5"}
    assert {row["reward_variant"] for row in stage_two} == {"ecp_v4"}
    assert manifest["total_updates"] == 6
