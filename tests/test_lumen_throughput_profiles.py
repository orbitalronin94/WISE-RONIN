from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_lumen_18_throughput.py"
SPEC = importlib.util.spec_from_file_location("lumen_throughput_tournament", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_tournament_has_exactly_18_unique_candidates():
    assert len(MODULE.CANDIDATES) == 18
    assert len(set(MODULE.CANDIDATES)) == 18


def test_candidate_deltas_only_change_execution_settings():
    for name, candidate in MODULE.CANDIDATES.items():
        MODULE.assert_execution_only(candidate["profile"])
        assert not (set(candidate["profile"]) & MODULE.PROTECTED_PROFILE_KEYS), name
        assert set(candidate.get("env", {})) <= {
            "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "KMP_BLOCKTIME"
        }, name


def test_no_candidate_disables_an_organ_or_changes_training_objective():
    forbidden = {
        "organs", "plasticity", "sleep_downscale", "sleep_interval_steps",
        "replay_compression", "delta_parallel_refinements", "delta_pre_normalize",
        "delta_reuse_write_update", "delta_always_apply_eligibility",
        "delta_trace_norm_diagnostics_on_demand", "delta_share_causal_masks",
        "homeostasis_fused_scatter", "no_sleep", "no_jepa", "no_rep",
    }
    for name, candidate in MODULE.CANDIDATES.items():
        assert not (set(candidate["profile"]) & forbidden), name
