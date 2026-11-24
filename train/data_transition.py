"""Explicit, auditable corpus-change handling for exact checkpoint resumes."""


def resolve_data_transition(saved_hash, new_hash, *, allow_change=False,
                            expected_parent_hash=None, step=-1,
                            label="explicit corpus transition"):
    if not saved_hash or not new_hash:
        raise ValueError("Both saved and new corpus SHA-256 values are required")
    if saved_hash == new_hash:
        return None
    if not allow_change:
        raise ValueError(
            "Checkpoint training corpus hash mismatch; pass "
            "--allow-data-change-on-resume only for an intentional corpus transition"
        )
    if expected_parent_hash != saved_hash:
        raise ValueError(
            "Explicit corpus transition parent hash does not match checkpoint: "
            f"expected_parent={expected_parent_hash}, checkpoint={saved_hash}"
        )
    return {
        "from_sha256": saved_hash,
        "to_sha256": new_hash,
        "at_step": int(step),
        "label": str(label),
        "optimizer_state_preserved": True,
        "tokenizer_must_remain_unchanged": True,
    }
