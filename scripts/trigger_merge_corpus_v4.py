"""Explicit trigger marker for the corpus-v4 freeze workflow.

The freeze workflow performs all downloads, merge checks and independent audits.
This file intentionally has no side effects when imported or executed.
"""

if __name__ == "__main__":
    print("Corpus v4 freeze workflow trigger marker; no training is started here.")

# Re-run freeze after both immutable candidate builders reached their exact quotas.
