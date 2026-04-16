def test_build_synthetic_snapshot_wraps_metadata_and_rows():
    from benchmarks.run_synthetic import build_synthetic_snapshot

    rows = [
        {
            "scenario": "A: R=3, L=500, phi=0.0",
            "method": "SCT",
            "E_r": 0.01,
            "coverage_90": 0.9,
            "coverage_80": 0.8,
            "width_90": 3.2,
            "width_80": 2.5,
            "pit_ks_p": 0.25,
            "resets": 15,
            "warm_start_occ": 0.42,
            "wall_seconds": 20.3,
        }
    ]

    snapshot = build_synthetic_snapshot(
        results=rows,
        n_replicates=10,
        seed_base=20260306,
        pit_seed=20260306,
    )

    assert snapshot["script"] == "benchmarks/run_synthetic.py"
    assert snapshot["n_replicates"] == 10
    assert snapshot["seed_base"] == 20260306
    assert snapshot["pit_seed"] == 20260306
    assert snapshot["n_rows"] == 1
    assert snapshot["results"] == rows
