from train.train import should_reuse_cached_gradient, should_surprise_gate


def test_surprise_gate_is_inclusive_and_optional():
    assert should_surprise_gate(1.8, 1.8)
    assert should_surprise_gate(1.2, 1.8)
    assert not should_surprise_gate(1.8001, 1.8)
    assert not should_surprise_gate(1.0, None)


def test_lazy_backward_reuses_only_alternating_steps_after_cache_exists():
    cached = [object()]
    assert not should_reuse_cached_gradient(True, 0, cached)
    assert should_reuse_cached_gradient(True, 1, cached)
    assert not should_reuse_cached_gradient(True, 2, cached)
    assert not should_reuse_cached_gradient(True, 1, None)
    assert not should_reuse_cached_gradient(False, 1, cached)
