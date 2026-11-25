"""Explicit trigger marker for the corpus-v4 freeze workflow.

The freeze workflow performs all downloads, merge checks and independent audits.
This file intentionally has no side effects when imported or executed.
"""

if __name__ == "__main__":
    print("Corpus v4 freeze workflow trigger marker; no training is started here.")

# Re-run freeze after both immutable candidate builders reached their exact quotas.

# Trigger freeze again after correcting the v3.2.005 manifest's optional tokenizer field.

# Re-run after distinguishing inherited v3.2 provenance from newly admitted v4 provenance.

# Re-trigger only after the freeze workflow is corrected for the 180K quota and guarded concurrency.
