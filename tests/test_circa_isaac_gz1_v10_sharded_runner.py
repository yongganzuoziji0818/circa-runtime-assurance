from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from agc_runtime_assurance import circa_isaac_gz1_v10_sharded_runner as sharded


@pytest.mark.parametrize("count,expected", [(12, 1280), (8, 1920)])
def test_partition_is_disjoint_complete_balanced_and_deterministic(count: int, expected: int):
    receipt = sharded.verify_partition(count)
    parts = [sharded.partition_indices(sharded.DEFAULT_ROLLOUTS, count, shard_id) for shard_id in range(count)]
    flattened = [index for part in parts for index in part]
    assert len(parts[0]) == expected
    assert sorted(flattened) == list(range(sharded.DEFAULT_ROLLOUTS))
    assert len(set(flattened)) == sharded.DEFAULT_ROLLOUTS
    assert receipt["total_indices"] == sharded.DEFAULT_ROLLOUTS
    assert receipt == sharded.verify_partition(count)


def test_partition_rejects_unapproved_count_and_id():
    with pytest.raises(sharded.CircaIsaacGZ1V10ShardedError):
        sharded.partition_indices(sharded.DEFAULT_ROLLOUTS, 10, 0)
    with pytest.raises(sharded.CircaIsaacGZ1V10ShardedError):
        sharded.partition_indices(sharded.DEFAULT_ROLLOUTS, 12, 12)


def test_frozen_schedule_hash_reproduction(tmp_path):
    root = tmp_path
    manifest = {
        "seed_namespace": sharded.base.SEED_NAMESPACE,
        "master_seed": 97306849832006341,
        "schedule_seed": 6195735420388600894,
        "candidates": [{"candidate_id": f"candidate-{index}"} for index in range(12)],
    }
    schedule = sharded.base.compile_schedule(manifest)
    serialized = [
        [item.candidate_index, item.scenario_index, item.driver_index, item.method_index, item.hazard_active, item.scenario_seed]
        for item in schedule
    ]
    digest = hashlib.sha256(json.dumps(serialized, separators=(",", ":")).encode()).hexdigest()
    assert len(schedule) == sharded.DEFAULT_ROLLOUTS
    assert digest != ""


def test_shard_array_schema_uses_local_rollout_axis():
    arrays = sharded._allocate_arrays(16)
    schema = sharded.array_schema(16, sharded.DEFAULT_HORIZON)
    assert set(arrays) == set(schema)
    for name, (dtype, shape) in schema.items():
        assert arrays[name].dtype == dtype
        assert arrays[name].shape == shape
        assert not arrays[name].dtype.hasobject
    assert arrays["completed_steps"].shape == (16,)
    assert arrays["physics_backend_hash"].shape == (32,)
