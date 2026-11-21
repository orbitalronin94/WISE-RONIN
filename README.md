# RONIN-LUMEN V1

RONIN-LUMEN V1 is the CPU-first refactor of WISE-RONIN. The previous WISE-RONIN V2 source tree is preserved unchanged under `archive/v2/`.

## Current status

**M1 skeleton only.** No V1 organ is implemented yet. The model shell exists only to verify shapes, CPU execution, and checkpoint/resume mechanics. Do not interpret its outputs as a trained language model.

## M1 smoke test

```bash
python -m pip install -r requirements.txt
python train/train.py --steps 10
python train/train.py --resume checkpoints/step_10.pt --steps 10
python train/train.py --steps 10 --output-dir /tmp/lumen-smoke
python eval/eval.py --holdout data/holdout_500.jsonl
```

The V2 holdout is archived unchanged at `archive/v2/holdout_wise_ronin_500.jsonl`. A copy for the future V1 evaluation path will be made only after its identity and hash are recorded; no holdout is used by the M1 dummy trainer.

## Next gate

M1 is not complete until the 10-step run, checkpoint creation, and resume run have been executed and verified. Work must stop after M1 pending owner approval.
