import pytest

from agc_runtime_assurance.aoi_cbf_author_eval import (
    AUTHOR_SEMANTICS,
    build_parser,
    parse_author_test_log,
    run_author_script_fidelity,
)


def test_parse_author_test_log(tmp_path):
    path = tmp_path / "test_log.csv"
    path.write_text("10,0.97,123.5,0.2,100,2.0,1.5,0\n", encoding="utf-8")
    assert parse_author_test_log(path) == {
        "num_agents": 10,
        "safe_rate": 0.97,
        "mean_length": 123.5,
        "mean_error": 0.2,
        "episodes": 100,
        "poisson_coefficient": 2.0,
        "communication_radius": 1.5,
        "outer_seed": 0,
    }


@pytest.mark.parametrize(
    "content",
    ["10,1,2\n", "10,1,2,3,4,5,6,7\n10,1,2,3,4,5,6,8\n"],
)
def test_parse_author_test_log_rejects_wrong_shape(tmp_path, content):
    path = tmp_path / "test_log.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        parse_author_test_log(path)


def test_wrapper_refuses_existing_output_before_external_imports(tmp_path):
    source = tmp_path / "source"
    logs = tmp_path / "logs"
    output = tmp_path / "evidence"
    source.mkdir()
    logs.mkdir()
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_author_script_fidelity(
            source_root=source,
            training_log_path=logs,
            output_root=output,
            outer_seed=0,
            episodes=100,
            num_agents=10,
            checkpoint_step=500000,
            gpu_index=0,
            expected_source_commit="a" * 40,
            timeout_seconds=1800,
        )


def test_cli_requires_explicit_evidence_paths_and_budget():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    assert AUTHOR_SEMANTICS == "author_test_py_delay_aware_false_predictor_not_loaded"
