# LUMEN — combined lazy backward + surprise gate (1,000 steps)

## Question

Can the compute savings from lazy backward and surprise gating be combined while
preserving or improving training loss versus the unmodified main training loop?

## Arms

- **Control:** unmodified LUMEN training loop, 1,000 steps.
- **Candidate:** one combined run, 1,000 steps:
  - Surprise gate: if scalar loss is <= 1.8, record the observation but skip
    backward, optimizer update and scheduler update.
  - Lazy backward: on alternate eligible steps, reuse the most recent full
    gradient multiplied by 0.9 instead of computing backward again.
  - The surprise gate takes precedence. A gated step does not count as a lazy
    gradient-reuse step.
  - Loss and affect/homeostasis diagnostics continue to include gated
    observations; the 100-step feedback update is not skipped.

## Fixed conditions

Both arms run sequentially on the same GitHub-hosted CPU runner, from the exact
LUMEN step-389700 checkpoint and restored optimizer/scheduler/RNG state, using
the same frozen corpus, tokenizer, token cache, seed (42), batch size and
sampled row sequence. The run order is control first, then candidate.

Parent artifacts come from run
https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45273897576

## Measurements

Each arm records mean loss, final-20 loss, first/last loss, end-to-end loop
seconds, steps/s, forward/backward/optimizer time and optimizer-update count.
The combined candidate additionally reports:
- surprise-gated steps;
- reused-gradient steps;
- total skipped backward calls;
- effective optimizer updates.

Timing excludes package installation, artifact downloads, checkpoint loading
and model initialization. It includes batch selection and the 1,000-step loop.

## Promotion gate

No automatic merge or production activation. Candidate qualifies for promotion
only if:
1. mean loss is strictly lower than control;
2. final-20 mean loss is no higher than control;
3. end-to-end elapsed time is strictly lower than control;
4. both surprise gating and lazy gradient reuse were actually exercised.

A result that fails one condition is not promoted. This is a 1,000-step training
microbenchmark, not a holdout or proof of long-horizon convergence. A positive
result must be followed by holdout and longer-run validation.
