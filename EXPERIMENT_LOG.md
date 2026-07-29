# Experiment Log — ese651_project_redo (Phase 1 sim redo)

Running log of this solo redo, kept as I go rather than written retroactively. Each entry states a
hypothesis/goal before the work, then what was actually done and what the data showed — including
dead ends. See `/Users/joshthekorean/development/drone-project/METRICS_GUIDE.md` for how the metrics
referenced below should be read, and `../CLAUDE.md` in this repo for the full assignment/process
context this log is following.

Format per entry: **Goal → What I did → Result (real numbers) → Conclusion / next step.**

---

## 2026-07-29 — Phase 0: setup, and confirming the starting state

**Goal.** Before writing any code, verify exactly what's stub vs. complete in this repo (don't trust
the assignment description alone), and get sign-off on an implementation plan.

**What I did.** Read `CLAUDE.md` (this repo's) and `METRICS_GUIDE.md` in full. Explored the repo on
`sim-redo` directly (full contents of `ppo.py`, `rollout_storage.py`, `actor_critic.py`,
`on_policy_runner.py`, `wandb_utils.py`, `train_race.py`, `play_race.py`, `rsl_rl_ppo_cfg.py`,
`quadcopter_strategies.py`, `quadcopter_env.py`, `README.md`). Had a second pass critique the planned
PPO/reward design against that exact code before writing anything. Wrote up a plan and confirmed it
with Josh.

**Result.**
- Repo is clean on `sim-redo`, byte-identical to the pristine template except `CLAUDE.md` and the
  W&B project rename (`68ee11d`) — confirmed via `git diff --stat 1e87843 HEAD`.
- `ppo.py` currently **does not parse** (`IndentationError`) — the `update()` TODO's `for` loop body
  is empty, not even a `pass`. Everything else in that file (`__init__`, `act`, `process_env_step`,
  `compute_returns`, storage/optimizer setup) is complete.
- `rollout_storage.py` — fully implemented, including a working GAE(λ) in `compute_returns()` (this
  is optional-to-explore per the assignment, not a required TODO) and `mini_batch_generator()`.
  Confirmed the exact field order/shapes the minibatch generator yields (this mattered — see below).
- `quadcopter_strategies.py` — the three required TODOs (`get_rewards`, `get_observations`,
  `reset_idx`) are exactly as described: hardcoded to spawn 2m behind gate 0, hover there, isotropic
  distance-threshold "gate pass" with no directionality. Found an unused buffer,
  `_prev_x_drone_wrt_gate` (declared, reset to `1.0`, never read anywhere) — this looks like the
  template author's intended hook for a sign-change gate-crossing test, currently dormant.
- `_n_gates_passed` is declared and zeroed but **never incremented** anywhere in the stub.
- `quadcopter_env.py`'s `observation_space = 1` is an explicit dummy placeholder (comment: "just
  needs to exist for Gymnasium compatibility") — doesn't need to track the real obs vector size.
- The known W&B "Crashed" bug is confirmed present and unfixed: `WandbSummaryWriter.stop()` exists
  (`wandb_utils.py`) but nothing calls it anywhere in the repo.
- Resolved three PPO-implementation ambiguities by reading `rollout_storage.py` directly rather than
  guessing: (1) buffer-sourced tensors (`values`/`returns`/`advantages`/`actions_log_prob`) carry a
  trailing singleton dim `(batch,1)` while the actor-critic's fresh outputs are `(batch,)` — mixing
  them directly would silently broadcast wrong instead of erroring; (2) `value_targets` in the
  `update()` tuple maps to old (collection-time) `values`, `discounted_returns` maps to `returns`,
  confirmed via the generator's exact yield order, not inferred from the names; (3) advantage
  normalization is already fully handled inside `compute_returns()` (gated by
  `normalize_advantage_per_mini_batch`, already `True` in effect for this config) — `update()` must
  not re-normalize.
- Per TA guidance (relayed by Josh, independent of anything in the sibling repos), read
  [arXiv 2406.12505](https://arxiv.org/abs/2406.12505) ("Demonstrating Agile Flight from Pixels
  without State Estimation") directly for its reward structure:
  `r = r_prog + r_perc + r_pass − r_cmd − r_crash`. Notably, the `r_prog` term
  (`λ1·(d_{t−1}−d_t)`, a potential-based progress delta) is almost exactly what I'd already
  independently planned to replace the stub's static closeness bonus with — good convergent
  validation. `r_perc` (rewards keeping the gate centered in a camera's optical axis) doesn't apply
  here since this assignment's observations are privileged state, not raw pixels — there's no
  camera-pointing problem to reward, which is a mechanistic reason for the TA's advice to skip it,
  not just a rule to follow blindly.

**Conclusion / next step.** Plan approved: implement PPO first and validate against the
unmodified-stub "hover near gate 0" checkpoint (Section 2, isolates the algorithm from any strategy
design), then redesign reward/observations/reset in isolated sub-steps (gate-pass detection →
progress reward → observations → reset randomization → full run), keeping the initial reward
deliberately minimal per the TA's advice (`r_prog` + `r_pass`, existing crash/death-cost plumbing;
no command-smoothness term until later, and only if flight actually looks jittery). Domain
randomization of the dynamics parameters is deferred to its own later ablation rather than bundled
in up front. Full reasoning in `/Users/joshthekorean/.claude/plans/humming-wondering-adleman.md`.

Also did a VM sanity check: `instance-20260302-161614` was stopped (correctly, from a prior
session). Started it, confirmed `env2`'s Python 3.10.20 at `~/anaconda3/envs/env2` and `~/IsaacLab`
both present as expected, cloned `ese651_project_redo` (`sim-redo` branch, public repo, no auth
needed) as a sibling to `IsaacLab` at `~/ese651_project_redo`. Noted the VM's non-interactive SSH
shells don't source `.bashrc` at all (it has the standard Debian early-return-if-not-interactive
guard right at the top) — not just a `WANDB_API_KEY` quirk as `CLAUDE.md` frames it, but a general
one: `conda activate` won't work either over `gcloud compute ssh --command=`; use full binary paths
(`~/anaconda3/envs/env2/bin/python`) or explicit `export`s instead. Stopped the VM again immediately
after — no training happens until PPO is actually implemented, so no reason to leave it running.

---

## 2026-07-29 — Phase 1a: PPO `update()` design, before writing any code

**Goal.** Implement the PPO-clip update step correctly on the first attempt, informed by the exact
scaffolding already in this repo rather than a generic textbook version that might not match this
codebase's conventions (variable names in the TODO were deliberately changed from upstream `rsl_rl`,
so I can't assume this is vanilla).

**Design (see the approved plan for the fuller version; this is the log copy of record).**

- **Shapes.** `rollout_storage.py` stores `values`/`returns`/`advantages`/`actions_log_prob` with an
  explicit trailing singleton dim; after `mini_batch_generator`'s `.flatten(0,1)` and indexing they
  arrive as `(batch, 1)`. But `ActorCritic.get_actions_log_prob()` and `.entropy` return `(batch,)`
  (they do `.sum(dim=-1)` over the action dimension internally). If I compute
  `new_log_prob - prev_log_probs` without reconciling this, PyTorch broadcasts `(batch,) - (batch,1)`
  into `(batch, batch)` — no error, just silently wrong gradients. Fix: `.squeeze(-1)` on
  `prev_log_probs`, `value_targets`, `advantage_estimates`, `discounted_returns` right after
  unpacking, before any arithmetic touches them.
- **Field identities**, confirmed from `mini_batch_generator`'s actual yield order (not the TODO's
  variable names alone, which are a red herring for `value_targets` vs `discounted_returns` — both
  *sound* like plausible names for either quantity): position 4 (`value_targets`) ← the buffer's
  `values` (old, collection-time critic output — the clip *reference*), position 6
  (`discounted_returns`) ← the buffer's `returns` (the actual GAE regression target). Getting these
  swapped wouldn't crash, it would just silently stop being PPO2 value-clipping.
- **Advantage normalization**: `PPO.compute_returns()` (already complete, not something I'm touching)
  calls `storage.compute_returns(..., normalize_advantage=not normalize_advantage_per_mini_batch)`,
  and `RolloutStorage.compute_returns()` does the actual global mean/std normalization internally
  when that flag is set. So `advantage_estimates` arriving in `update()` is already normalized — I
  must not normalize it again.
- **Detach**: `PPO.act()` already calls `.detach()` on everything it stores into the transition
  (log-probs, values, action mean/std). So nothing coming out of the generator needs an extra
  `.detach()` — but the *new* forward pass (recomputing log-prob/value/entropy under the current
  policy) must **not** be wrapped in `no_grad()`, since gradients need to flow through it.
- **Per-minibatch computation**:
  1. Re-run the current policy on `observations`/`critic_observations` → fresh log-prob (for
     `sampled_actions`), value, mean/std, entropy.
  2. Analytic Gaussian KL between old (`prev_mean_actions`, `prev_action_stds`) and new params,
     **summed over the action dimension then averaged over the batch** (matching the same
     `.sum(dim=-1)` convention `get_actions_log_prob`/`entropy` already use — averaging over the
     action dim instead would silently rescale KL and miscalibrate `desired_kl=0.01`). Computed
     under `torch.no_grad()` since it's LR bookkeeping, not part of the loss graph. Since
     `schedule="adaptive"`: nudge `self.learning_rate` down if KL is well above `desired_kl`, up if
     well below, clamp to sane min/max, write into `optimizer.param_groups[...]['lr']` before this
     batch's step.
  3. Clipped surrogate: `ratio = exp(new_log_prob - prev_log_probs)`,
     `loss = -mean(min(ratio*A, clip(ratio, 1±clip_param)*A))`.
  4. Clipped value loss (`use_clipped_value_loss=True`): clip `(value - value_targets)` to
     `±clip_param`, take the max of clipped/unclipped squared error against `discounted_returns`.
  5. Entropy bonus subtracted, coefficient `entropy_coef=0.0` today so numerically inert — still
     wiring it correctly since it's logged (`Loss/entropy`) and tunable. Flagging now, before any
     run: with zero entropy coefficient nothing is fighting premature action-std collapse, so if the
     std collapses before ~iteration 100 (the metrics guide's red flag), the fix to try first is a
     small positive `entropy_coef` (e.g. 0.005), not an assumption that the reward is broken.
  6. `zero_grad()` → `backward()` → `clip_grad_norm_(max_grad_norm)` → `step()`. Accumulate
     `mean_value_loss`/`mean_surrogate_loss`/`mean_entropy` via `.item()`, not raw tensors (avoids
     holding each minibatch's graph in memory for the rest of the loop).
- **Also fixing while in this file's neighborhood**: wiring `WandbSummaryWriter.stop()` into
  `train_race.py` after `runner.learn()` — it's defined but never called anywhere in the repo, which
  is why every run shows "Crashed" on the W&B dashboard regardless of outcome.

**Result.** Implemented `update()` per the design above (`ppo.py` lines 148-220). Confirmed with
`ast.parse()` (a free, local, zero-GPU check) that the file parses correctly now — it did not before
(the empty `for` body was an `IndentationError`, so nothing importing this module would even have
run). While fixing the W&B `stop()` bug, checked `on_policy_runner.py` directly rather than assuming
`runner.writer.stop()` is always safe to call: the default (non-`--logger wandb`) path constructs a
plain `torch.utils.tensorboard.SummaryWriter`, which has no `.stop()` method at all — only
`WandbSummaryWriter`/`NeptuneSummaryWriter` define it. Calling it unconditionally would have broken
every default-logger run (`AttributeError`) to fix a bug that only affects the wandb path. Guarded it
with `hasattr(runner.writer, "stop")` instead.

**Conclusion / next step.** Both changes are in and pass a local syntax check, but neither has
actually *run* yet — that requires the VM. Next: push this to `origin/sim-redo`, then a small
(~64–256 env, ~10–20 iteration) smoke test using the *unmodified* stub reward/obs/reset, checking:
no crash/NaN, surrogate loss small-and-noisy (not pinned at exactly 0), value loss trending down,
action-noise std actually moving off its 1.0 init. Checking in with Josh before that run — first
real GCP compute use this session.

---

## 2026-07-29 — Phase 1b: PPO smoke test (unmodified stub reward)

**Goal.** Cheap confirmation that `update()` actually runs correctly on real hardware before
spending any real compute on a full checkpoint run — catch shape bugs, NaNs, or a hung process while
it's still nearly free.

**What I did.** `num_envs=256`, `max_iterations=20`, unmodified stub reward/obs/reset, `--logger
wandb` (run: https://wandb.ai/ese651/ese651_quadcopter_josh_redo/runs/4hmlqnk9). Ran on the VM via a
`nohup`'d background process over SSH so I could poll the log without holding the session open.

**Result.**
- Ran to completion (all 20 iterations) with no crash, no hang, no NaN/Inf anywhere in the log.
- `Surrogate loss`: 0.0143 → 0.0002 → ... → -0.0014 — small-magnitude and noisy around zero, not
  pinned at exactly 0 (would mean no gradient reaching the actor) and not blowing up. Matches what
  the metrics guide says to expect — this one isn't supposed to visibly "converge."
- `Mean action noise std`: 1.00 at the start, 0.99 by iteration 14 — moving in the right direction,
  just slowly, which is expected this early (the metrics guide's own reference run was still near
  1.0 at iteration ~27).
- `Episode_Reward/progress_goal`: 46.45 (iter 4) → 74.51 (iter 18) — climbing, as expected even with
  the hover-only stub reward.
- `Episode_Reward/crash`: stayed small (-0.004 to -0.17) throughout.
- `Value function loss`: 17k → 30k, i.e. **not** trending down within just 20 iterations — flagged
  as a real open question, not brushed aside: could be normal (not remotely enough iterations for the
  critic to converge, especially while the adaptive-KL schedule was simultaneously ramping the
  learning rate up toward its ceiling, meaning the policy — and therefore the value target
  distribution — was still shifting fast), or could indicate a real problem with the value loss
  wiring. Twenty iterations isn't enough evidence either way; watching this specifically in the
  longer Phase 1c run rather than guessing now.
- `Loss/learning_rate` ended at 0.01 — the adaptive-KL schedule's upper clamp. Confirms the schedule
  is actually executing and adjusting (not just inert code): with a near-random early policy, KL
  divergence per update is naturally small, so the "ramp up when below desired_kl/2" branch fired
  repeatedly until it hit the ceiling I set.
- Confirmed the W&B fix actually works end-to-end: the log shows `wandb: Run summary`, a full
  `Run history`, and `wandb: Synced 5 W&B file(s)...` — `wandb.finish()` ran and closed out the run
  cleanly before `simulation_app.close()`, so this shows as a finished run rather than "Crashed".
- Confirmed logging isolation: the run landed under `ese651/ese651_quadcopter_josh_redo`, not the
  shared `ese651_quadcopter` project.

**Conclusion / next step.** PPO passes the cheap smoke test — proceeding to Phase 1c, a longer run
(`num_envs=2048`, `max_iterations=200`, still the unmodified stub reward) to actually reach the
assignment's "hover near gate 0" checkpoint and get a real read on whether the value loss trend from
this run was just short-horizon noise.

---

## 2026-07-29 — Phase 1c: longer PPO checkpoint run (still unmodified stub reward)

**Goal.** Reach the assignment's actual Section-2 checkpoint — a policy that hovers near gate 0 —
and get enough iterations to tell whether the smoke test's inconclusive value-loss trend was real or
just short-horizon noise.

**What I did.** `num_envs=2048`, `max_iterations=200`, unmodified stub reward, `--logger wandb`.
Run: https://wandb.ai/ese651/ese651_quadcopter_josh_redo/runs/dzl2jjhd

**Result — the good signs.** Every *task*-level metric moved in the right direction, clearly and
monotonically, over the full 200 iterations:
- Mean total reward: 1413 → 4784 (roughly 3.4x)
- `Episode_Reward/progress_goal`: 46.5 → 157.3
- `Episode_Reward/crash`: stayed small throughout (-0.11 to -0.23, no trend toward worse)
- Mean episode length: ~100 → ~256 steps (survives noticeably longer without dying)
- Throughput: ~28,000 steps/s at this scale (2048 envs) — cheap, ~350s of actual compute for the
  full run

**Result — two things I'm not immediately certain about, logged honestly rather than glossed over:**
1. **Mean action noise std did not shrink** — 1.00 at init, still ~1.02–1.03 at iteration 199. Per
   the metrics guide this is normally a red flag ("doesn't shrink → policy never commits to
   confident behavior"). But my first instinct going in — "if std collapses too early, add
   `entropy_coef`" (logged in the Phase 1a entry above) — turns out to be the fix for the *opposite*
   problem. This is std failing to shrink at all, and adding entropy_coef would push the *wrong*
   direction (a positive entropy coefficient rewards staying stochastic, which would suppress
   shrinkage further, not encourage it). That earlier contingency plan doesn't apply here; I need a
   fresh hypothesis for this actual failure mode.
2. **Value function loss did not clearly trend down** — oscillating in the 17k–30k range across all
   200 iterations. In isolation this also reads as a red flag per the guide.

**Working hypothesis, and why I'm not treating either as a confirmed bug yet.** Both anomalies may
be the *same* underlying story rather than two separate problems: if the critic isn't fitting well
(hence the flat/noisy value loss), the advantage estimates it produces are correspondingly noisy,
which would directly explain why the policy isn't getting a clean, consistent gradient signal to
confidently narrow its action distribution (hence flat std). And the more parsimonious explanation
for *why* the critic hasn't converged yet: every task-relevant metric (reward, progress_goal,
episode length) was **still visibly, steeply climbing** at iteration 199, not plateaued — this run
was cut off while clearly still in an early, fast-improving phase, and `Episode_Termination/time_out`
sat at exactly 0.0 the entire run (episodes are *always* ending via early death, never surviving to
the full 30s/1500-step episode) — so a genuinely stable hover hasn't been reached yet either. Given
that, it would be premature to expect the "confidence" signals (shrinking std, converging value
loss) to have kicked in — those typically lag task performance, not lead it. I don't have strong
enough evidence to rule out a real bug, but "just needs more iterations" is the more likely
explanation given the trend, and it's also the cheapest hypothesis to test directly.

**Investigated one more thing before deciding how to extend the run**: whether the adaptive
learning rate would survive a `--resume`. Checked `on_policy_runner.py`'s `save()`/`load()` directly
rather than assuming. Finding: the *optimizer's* actual LR round-trips correctly (Adam's
`state_dict` includes `param_groups[...]['lr']`, and `load()` restores it). But `self.alg.learning_rate`
— the plain attribute my adaptive-KL code reads and writes every update, and which
`Loss/learning_rate` logs — is **not** persisted anywhere; `PPO.__init__` always re-seeds it from
the config's default (5e-4) on every process start, resume or not. So a `--resume`d run would start
its first update computing the adaptive adjustment from 5e-4, then immediately overwrite the
correctly-restored 0.01 optimizer LR with whatever that computes — silently clobbering the resumed
state with a discontinuity. Not fixing this now (it's not blocking anything and isn't needed to hit
Milestone 1), but noting it as a real, minor gap rather than letting it bite silently later if I
reach for `--resume` on a more expensive run.

**Conclusion / next step.** Given the resume-LR quirk above, extending via a clean *fresh* run
(same `num_envs=2048`, longer `max_iterations`) is more interpretable than resuming — no
discontinuity to account for when reading the curve. Launching `max_iterations=600` (3x this run)
next, same everything else, specifically to check: does `progress_goal`/reward growth start to
plateau, does episode length approach the full 1500-step max (a real stable hover), and — the actual
test of the hypothesis above — do action-std and value loss start moving in the expected direction
once task performance itself levels off.

---
