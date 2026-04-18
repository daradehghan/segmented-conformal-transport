import json
from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "state_schema_0_1_0"


def test_pinned_state_schema_fixture_loads():
    from tsconformal import load_calibrator

    calibrator = load_calibrator(FIXTURE_DIR)

    assert calibrator.segment_id == 0
    assert calibrator.num_resets == 0
    assert calibrator.last_reset_t == 0
    assert calibrator._grid.tolist() == pytest.approx(
        [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0]
    )


def test_unsupported_state_schema_versions_are_rejected(tmp_path):
    from tsconformal import load_calibrator

    state = json.loads(FIXTURE_DIR.joinpath("meta.json").read_text(encoding="utf-8"))
    state["schema_version"] = "0.2.0"

    fixture_dir = tmp_path / "unsupported-schema"
    fixture_dir.mkdir()
    fixture_dir.joinpath("meta.json").write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported state schema version"):
        load_calibrator(fixture_dir)


def test_serialized_state_uses_schema_constant_not_package_version(
    tmp_path,
    monkeypatch,
):
    import tsconformal
    from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator, save_calibrator
    from tsconformal.calibrators import STATE_SCHEMA_VERSION

    monkeypatch.setattr(tsconformal, "__version__", "9.9.9", raising=False)

    calibrator = SegmentedTransportCalibrator(
        grid_size=5,
        rho=0.95,
        n_eff_min=2.0,
        step_schedule=lambda n: 0.1,
        detector=CUSUMNormDetector(),
        cooldown=2,
        confirm=1,
    )
    output_dir = tmp_path / "saved-state"
    save_calibrator(calibrator, output_dir)

    saved_state = json.loads(output_dir.joinpath("meta.json").read_text(encoding="utf-8"))
    assert saved_state["schema_version"] == STATE_SCHEMA_VERSION
    assert saved_state["schema_version"] != tsconformal.__version__
