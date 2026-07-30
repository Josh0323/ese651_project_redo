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

## 2026-07-29 — Milestone 1: PPO validated (600-iteration run + video review)

**Goal.** Resolve the two open questions from the 200-iteration run (flat action-std, non-converging
value loss) by testing the "just needs more iterations" hypothesis directly, then confirm the
resulting behavior actually looks like hovering, not just scores well.

**What I did.** Same config as Phase 1c, `max_iterations=600` (fresh run, not resumed — see the LR
gap noted above). Run: https://wandb.ai/ese651/ese651_quadcopter_josh_redo/runs/3t7dykbe. Then
loaded `best_model.pt` from this run into `play_race.py` (`--num_envs 1 --video --video_length 300`),
downloaded the resulting video, and reviewed extracted frames directly rather than trusting the
metrics alone.

**Result — hypothesis confirmed, with real numbers across the full trajectory:**

| | Smoke test (20 it) | Phase 1c (200 it) | Extended (600 it) |
|---|---|---|---|
| Mean total reward | 1413 → 2174 | → 4784 | → **47,629–50,612** (peak 60,410 @ it 550) |
| `Episode_Reward/progress_goal` | 46.5 → 74.5 | → 157.3 | → **~1,489–1,985** |
| Mean episode length (steps, max 1500) | ~100–116 | → ~256 | → **~1,249–1,302** |
| `Episode_Termination/time_out` | 0 | 0 (always died first) | **1.04–1.58** (now the dominant ending) |
| `Episode_Termination/died` | 2.4–2.9 (of 256 envs) | 8.1–9.0 (of 2048 envs) | **0.04–0.42** (of 2048 envs) |
| Mean action noise std | 1.00 → 0.99 | → 1.02–1.03 (flat/up) | → **0.82**, clearly shrinking |
| Value function loss | 17k → 20k | → 21k–30k (noisy) | mid-run 68.7k, **final 2.8k–5.4k** |
| `Loss/learning_rate` (adaptive) | rising toward ceiling | hit ceiling (0.01) | **0.00038** — schedule throttled itself back down |

Every one of the ambiguous signals from the 200-iteration run resolved itself with more training,
exactly as the working hypothesis predicted: episodes now overwhelmingly end via `time_out` instead
of `died` (a real behavioral shift, not just a bigger number), action-std is now visibly and
monotonically shrinking, and value loss dropped by roughly an order of magnitude once task
performance itself stopped changing so fast under it. The adaptive-KL learning-rate schedule's own
trajectory (ramp up early when updates are "cheap" against a near-random policy → throttle down hard
as the policy commits to specific behavior) is itself a nice confirmation that the mechanism I
implemented is self-regulating correctly, not just inert code that happens not to crash.

**Result — the video review caught something the numbers alone didn't make obvious.** The policy
*is* genuinely stable — frames sampled across the full 6-second clip show the drone parked in almost
exactly the same spot the whole time, matching the near-zero death rate. But that spot is **tucked
into a corner of gate 0's frame, not centered on the goal marker or anywhere near the actual 1×1m
opening**. This is a clean, concrete illustration of exactly what the assignment means by "the stub
reward will not produce a racing policy": `progress = 1 - tanh(distance/3)` saturates quickly, so
once the drone is "close enough," there's very little additional reward gradient pulling it the rest
of the way to the true goal position — and if drifting closer carries any risk (of a contact-sensor
hit near the frame, say), the policy has no reason to take it. This isn't a bug in my PPO
implementation; it's the expected, honest failure mode of a distance-only reward with no explicit
notion of "passed through the opening," which is precisely what Section 3 needs to fix.

**Checkpoint criteria from the assignment, assessed explicitly rather than assumed:**
- "`progress_goal` climbing" — yes, clearly and substantially (46 → ~1,900+).
- "`crash` staying near zero" — yes, throughout (-0.003 to -0.23, no worsening trend).
- "`gate_pass` never firing" — **not actually checkable on this stub**: the current reward dict only
  has `progress_goal_reward_scale`/`crash_reward_scale`/`death_cost`, so there's no `gate_pass`
  reward component or `Episode_Reward/gate_pass` metric logged at all yet (and `_n_gates_passed` is
  never incremented in the stub, confirmed back in the Phase 0 entry). This criterion is vacuously
  true right now rather than something I actually verified — worth being precise about rather than
  claiming a check I didn't really perform.

**Conclusion.** PPO's `update()` implementation is validated: mechanically correct (no crashes, sane
losses, all three return values wired and consumed correctly by the runner/logger), and it produces
the exact qualitative behavior change over training that the theory predicts (KL-adaptive LR
self-regulating, exploration narrowing as confidence grows, episodes surviving longer as the policy
improves). This is **Milestone 1**. Moving on to Section 3 (reward/observation/reset redesign) next
— and the corner-hovering finding above directly motivates the first concrete design change: a
gate-pass condition needs to actually verify passage through the opening, not just proximity to the
gate.

---

## 2026-07-29 — Phase 2a design: gate-pass detection + `_n_gates_passed`

**Goal.** Replace the stub's `dist_to_gate < 0.1` isotropic check with a directional test that
actually verifies passage through the 1×1m opening, and start incrementing `_n_gates_passed` (never
touched in the stub). Deliberately *not* touching the progress reward or reset randomization in this
step — isolating this one change so a smoke test afterward has exactly one thing that could explain
whatever happens.

**Design.**
- **Mechanism**: `_pose_drone_wrt_gate[:, 0]` is the drone's position along the *current target
  gate's own local forward axis* (confirmed directly from `quadcopter_env.py::_setup_scene` —
  `_normal_vectors` and `_waypoints_quat` are built from the same `scipy` rotation object, so
  whatever "local +x" means for one is exactly what it means for the other). The stub already
  declares and resets a `_prev_x_drone_wrt_gate` buffer (init `1.0`) that's never read anywhere —
  a strong signal this is the template author's intended hook for a sign-change test. Confirmed the
  sign convention concretely rather than assuming it: `reset_idx` spawns the drone at
  `gate_pos + 2·gate_normal_direction` (worked through the actual rotation arithmetic in
  `reset_idx`'s local→world transform), i.e. on the **positive**-local-x side, matching
  `_prev_x_drone_wrt_gate`'s `+1.0` init. So a forward pass through the gate is
  `x_prev > 0` transitioning to `x_now <= 0` — and I don't have to independently derive or guess a
  yaw→forward sign convention by hand to get this right, which is exactly the kind of thing that's
  easy to get backwards (confirmed for myself that naively using `(cos ψ, sin ψ)` from yaw directly,
  without checking it against this reset-position arithmetic, would *not* obviously have been safe
  to assume).
- **Opening bounds**: `torch.abs(y) < half_extent` and `torch.abs(z) < half_extent` in the same
  gate-local frame, so it counts as passing through the opening rather than anywhere on the infinite
  gate plane. Derived `half_extent` from `gate_model.gate_side / 2 * 0.8` (an 80%-of-true-half-width
  margin) rather than hardcoding a number independent of the actual configured gate size — the gates
  have real collision geometry already (confirmed in the Phase 0 entry), so this is about rejecting
  "technically crossed the plane but nowhere near the actual opening," not about collision safety,
  which the contact sensor already handles separately.
- **The subtlety that actually took the most thought**: `_prev_x_drone_wrt_gate` has to be updated
  every step for the *next* step's comparison — but on the exact step a pass is detected, `_idx_wp`
  advances to the *next* gate, so next step's `_pose_drone_wrt_gate` will already be computed
  relative to the new target. If I naively cache this step's (old-gate-frame) `x` value as
  `_prev_x_drone_wrt_gate`, the very next step would compare a new-gate-frame `x_now` against an
  old-gate-frame `x_prev` — two unrelated numbers, not a meaningful sign change, for one step per
  gate pass. This is the exact same "reference changes discontinuously when the target switches"
  problem as the progress-reward delta (planned for 2b), just showing up in the crossing-detector
  instead. Fix: for envs where a pass was just detected, immediately recompute their gate-relative
  pose against the *new* `_idx_wp` before caching it into `_prev_x_drone_wrt_gate`, mirroring what
  `reset_idx` already does when spawning fresh into a gate's frame — non-passing envs just carry
  forward this step's value as normal.
- **Gate-pass reward**: adding a `gate_pass` component, scaled by proximity to the gate center at
  the moment of passing (`1.0 - sqrt(y² + z²)`, clamped at 0) rather than a flat bonus — per the
  TA-recommended paper's `r_pass` term, and it directly serves the "quality of pass, not just count"
  concern `METRICS_GUIDE.md` already flags. Deliberately keeping the *progress* reward term
  completely unchanged in this step (still the stub's `1-tanh(distance/3)` form) — 2b is where that
  gets redesigned, and changing both at once would make a regression in the smoke test ambiguous
  about which change caused it.
- **Scale**: `gate_pass_reward_scale = 100.0` as a first guess, not a derived value — roughly "a
  clean pass is worth about as much as 2 steps of maximum progress reward," chosen to be assertive
  enough to matter against the existing `death_cost=-10.0` risk of attempting a pass near the frame,
  but this is exactly the kind of number I expect to have to tune once I can see whether gate-passing
  actually starts happening at all.

**Conclusion / next step.** Implement in `quadcopter_strategies.py::get_rewards()`, add
`gate_pass_reward_scale` to `train_race.py`'s reward-scale dict, syntax-check locally, then smoke
test on the VM (still fixed gate-0 spawn) watching specifically for: does `Episode_Reward/gate_pass`
ever go nonzero, does `_n_gates_passed` (not directly logged yet, but inferable from `_idx_wp`
behavior) actually advance, and does the sign-change logic avoid any obviously spurious behavior
right after a pass.

**Implemented** (`quadcopter_strategies.py::get_rewards()`, `train_race.py`). Both files pass a
local syntax check. Not yet run — bundling with Phase 2b below and smoke-testing both together,
since 2b's change is small enough that testing them as one pass is reasonable, but see 2b's entry
for why I still kept them as logically separate diffs.

---

## 2026-07-29 — Phase 2b design: delta-distance progress reward

**Goal.** Replace the stub's `1 - tanh(distance/3)` closeness bonus with a potential-based *delta*
that rewards actually closing distance to the gate — this is the direct fix for the Milestone-1
video finding (a policy that's "close enough" under the old form has very little gradient left
pulling it the rest of the way to the goal).

**Design.**
- `progress = last_distance_to_goal - distance_to_goal_now` (positive when the drone got closer this
  step). This is exactly the `r_prog` form from the TA-recommended paper (`λ1·(d_{t-1}-d_t)`) — and
  matches what I'd already planned independently before reading it, which I take as a good sign
  rather than a coincidence to worry about.
- **Reused existing dead plumbing, again.** `_last_distance_to_goal` is already declared and set in
  `reset_idx` — but like `_prev_x_drone_wrt_gate` before Phase 2a, it's never actually *read*
  anywhere in the stub. Second instance of the same pattern: scaffolding pre-wired for a design the
  stub's actual reward logic never got around to using.
- **One small correction while wiring it up**: `reset_idx` currently sets `_last_distance_to_goal`
  from only the XY components (`[:, :2]`), while the `distance_to_goal` already computed elsewhere in
  `get_rewards()` (for the old closeness bonus) is full 3D. Since this buffer was dead code before,
  there's no established convention to preserve — I'm defining it as full 3D for consistency with
  the rest of the function (this track has real z-variation between gates, e.g. 0.75m vs 2.0m, so a
  2D-only progress signal would ignore altitude closing entirely). Touching this one line in
  `reset_idx` now, ahead of 2d's actual reset-randomization work, because it's a required companion
  to this change, not new scope — without it, the very first step after every reset would compute a
  3D-vs-2D-mismatched delta.
- **Same transition-discontinuity issue as Phase 2a, same fix**: on the step a gate is passed,
  `_desired_pos_w` (this step) and `_last_distance_to_goal` (last step) end up referencing different
  gates. Zeroing the progress term for exactly those envs on that one step, rather than trying to
  construct a "consistent" cross-gate distance measure that wouldn't actually mean anything.
- **Defensive clamp** (`±1.0`) on the per-step delta — not expected to ever bind at realistic speeds
  (even at 10 m/s and a 50 Hz control rate, a legitimate single-step distance change is on the order
  of 0.2 m), but cheap insurance against an edge case I haven't thought of rather than a load-bearing
  part of the design.
- **Scale left unchanged for now** (`progress_goal_reward_scale=50.0`), deliberately. The *form*
  change alone shifts the natural per-step magnitude quite a bit (bounded-motion deltas are much
  smaller than the old form's O(1)-every-step closeness value), so the "right" scale is now a
  genuinely open question — but changing the form and the scale in the same pass would make it
  impossible to tell which change was responsible for whatever the smoke test shows. Watching the
  actual `Episode_Reward/progress_goal` magnitude in the smoke test before touching the number again.

**Conclusion / next step.** Implement alongside 2a (small change, same file, and I want one smoke
test to cover both before spending more VM time) — but keeping them as clearly separable diffs/log
entries so if something looks wrong, there's a documented, isolated place to start looking rather
than one undifferentiated "Phase 2" change.

**Implemented** (`quadcopter_strategies.py`: `get_rewards()`'s progress calc + the companion 3D fix
in `reset_idx`). Both files pass a local syntax check. Spawn is still hardcoded to gate 0 at this
point — 2d is where that changes. Next: smoke test 2a+2b together on the VM, unmodified reset
(still fixed gate-0 spawn), watching for: `Episode_Reward/gate_pass` going nonzero at all, no
crash/NaN, and `Episode_Reward/progress_goal`'s new magnitude (expected to look different in scale
now that it's a bounded delta instead of an O(1)-every-step closeness value).

---

## 2026-07-29 — Phase 2a+2b smoke test (256 envs, 20 iterations, still fixed gate-0 spawn)

**Goal.** Cheap check that both new mechanisms actually run and behave sanely before spending more
compute — not looking for real learning progress at 20 iterations, just: does it crash, does
`gate_pass` fire at all, does `progress_goal`'s new form look like a bounded delta rather than the
old saturating closeness value.

**Result.** Ran clean, no crash/NaN/traceback.
- `Episode_Reward/gate_pass`: 0.30–0.68 (nonzero) — the sign-change + opening-bounds detector is
  actually firing, even under a still-mostly-random 20-iteration policy in 256 parallel envs. That's
  the main thing this smoke test needed to prove.
- `Episode_Reward/progress_goal`: now fluctuating, including negative values (-3.55 to -0.59) — a
  real, expected behavior change, not a red flag. A delta-based reward should average out close to
  zero (positive and negative in roughly equal measure) under an undirected/near-random policy,
  unlike the old closeness bonus which was structurally always positive regardless of motion
  direction. This is actually the more honest signal — right now the reward is correctly reporting
  "the policy isn't moving purposefully yet," which the old reward form couldn't distinguish from
  "the policy is doing fine."
- `Value function loss`: 441–600, a ~40x drop from the old reward's 17k–30k at the same point in
  training. Purely a scale artifact: the new progress term's per-step magnitude is bounded and much
  smaller than the old form's O(1)-every-step value, so returns (and their squared error) are
  naturally smaller. Not a sign the critic got better or worse, just a different unit scale to
  recalibrate expectations around.
- `Mean total reward`: negative (-21 to -64) for the same underlying reason as `progress_goal`.
- Termination/death rate, action-noise std: consistent with what earlier 20-iteration runs looked
  like — this smoke test isn't long enough to expect movement here yet.

**Conclusion.** Both mechanisms work end-to-end. Not running a longer validation pass on 2a+2b in
isolation — per the phased plan, the real substantive training run happens at 2e once observations
(2c) and reset randomization (2d) are also in place; each sub-phase before that just needs a cheap
"does it run correctly" check, not a convergence run. Moving to 2c.

---

## 2026-07-29 — Phase 2c design: observation redesign

**Goal.** Replace the stub's 13-dim, partly-world-frame observation vector with a fully egocentric
one, and add lookahead (next gate, not just current) for the powerloop/chicane sequences.

**Verified one API assumption before relying on it**, rather than guess: I wanted a body-frame
gravity-direction vector as the attitude representation (avoids the quaternion double-cover
ambiguity `q`/`-q`, no representational singularity, unlike the stub's raw world-frame quaternion).
Isaac Lab commonly exposes this as `projected_gravity_b` on `ArticulationData` in other tasks I'm
aware of, but I hadn't confirmed it exists in *this* installation — grepped
`~/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py` on the VM directly
and confirmed it's defined at line 760. (This grep run into the VM's SSH/IAP flakiness noted in
`CLAUDE.md` — took several retries and one full VM restart before landing cleanly; not a code issue,
just infra noise worth not over-reading into.)

**Design — final 19-dim vector, all body-frame/egocentric, nothing in world frame at all:**
- `drone_lin_vel_b` (3) — already used in the stub, kept.
- `drone_ang_vel_b` (3) — body rates; available (`root_ang_vel_b`) but unused in the stub, added.
- `projected_gravity_b` (3) — replaces the stub's raw `drone_quat_w`. Deliberate tradeoff: this is
  yaw-invariant by construction (only encodes roll/pitch), so yaw information is dropped from this
  component specifically — but body angular velocity (yaw rate) and the gate-relative bearing below
  already carry the directional information a policy needs, so this isn't an accidental loss.
- `current_gate_pos_b`, `next_gate_pos_b` (3 each, 6 total) — gate position expressed in the
  **drone's** body frame (`subtract_frame_transforms(drone_pos, drone_quat, gate_pos_w)`), not the
  gate's own frame. This is a deliberate switch from `_pose_drone_wrt_gate` (drone-in-gate-frame,
  which Phase 2a's reward/gate-pass logic uses and keeps using unchanged) to gate-in-drone-frame for
  the *observation* specifically — "target is to my left/ahead/above" is a more directly
  action-relevant signal for a reactive control policy than "I am right/behind/below the target's
  own reference frame," even though the two are related by a fixed rotation and carry equivalent
  information in principle. The stub's own commented-out hint (`gate_pos_b, _ =
  subtract_frame_transforms(...)`) already pointed at this — another instance of unused scaffolding
  suggesting the intended design. Adding the **next** gate (not just current) specifically because
  the powerloop and chicane are sequential maneuvers where knowing what's coming after the immediate
  target seems likely to matter for approach planning — this is a hypothesis, not a certainty; if it
  turns out not to help, it's a cheap thing to ablate later.
- `_previous_actions` (4) — already computed and maintained by the env, just not previously exposed
  to the policy; added directly.
- **Deliberately dropped entirely**: world-frame position and world-frame quaternion. The track
  layout is fixed, so there's no generalization argument for absolute position the way there would
  be on a randomized-layout task — and gate-relative vectors already supply complete navigation
  information without it. Not including a gates-passed counter either, at least for now — the reward
  doesn't have any lap-aware shaping yet, so there's no clear task-relevant use for it, and adding
  observation dimensions without a concrete reason to is exactly the kind of thing worth resisting
  per the TA's simplicity advice, even though that advice was originally about the reward.

**On validation**: the plan called for checking shape/wiring "at zero GPU cost" before spending
compute — but Isaac Sim can't run at all without the VM (no local fallback on the Mac), and
`OnPolicyRunner` sizes the actor/critic dynamically from whatever `get_observations()` actually
returns at runtime (confirmed back in the Phase 0 exploration — `observation_space=1` in the env cfg
is a real no-op placeholder, not something that has to match). So the cheapest real validation is
just a very short VM run (small `num_envs`, a couple of iterations) rather than a separate synthetic
shape-check script — if construction fails, it fails immediately, before any real training time is
spent either way.

**Conclusion / next step.** Implement, syntax-check locally, then a small VM run.

**Result.** `num_envs=64, max_iterations=5` on the VM: ran clean, no shape/construction errors — the
actor/critic sized itself correctly against the new 19-dim vector with no changes needed elsewhere.
`gate_pass` and `crash` both read exactly `0.0000` the whole run, which is expected rather than
suspicious at this scale: episodes are only lasting ~60–75 steps here, and the crash penalty is
gated to not count until `episode_length_buf > 100` (existing stub logic, unchanged) — it literally
cannot have fired yet. Moving to 2d.

---

## 2026-07-29 — Phase 2d design: reset randomization

**Goal.** Randomize starting gate, position, heading, and initial velocity instead of always
spawning fixed behind gate 0 — the last piece before a real full-track training run (2e).

**Design.**
- **Random starting gate**: `waypoint_indices` uniform over all 7 (`torch.randint`) instead of
  hardcoded zeros. Exposes the policy to the whole course from iteration 0 instead of only ever
  training near gate 0 and hoping it generalizes later.
- **Position**: kept the existing local-frame rotation math (it already correctly generalizes to any
  gate once `waypoint_indices` isn't hardcoded — didn't need to change that part), randomizing what
  were previously fixed constants: distance behind the gate `1.0–3.0m` (was fixed `2.0`), added
  lateral jitter `±0.7m` and height jitter `±0.3m` (previously zero for both). Height jitter is
  relative to *each gate's own* z, so it's safe against the env's altitude bounds regardless of
  which gate gets sampled (checked both gate heights in this track, 0.75m and 2.0m, against
  `min_altitude=0.1`/`max_altitude=3.0` — comfortable margin either way).
- **Checked one thing before assuming it still holds**: Phase 2a's gate-pass detector relies on
  `_prev_x_drone_wrt_gate` being reset to `+1.0`, on the assumption the drone spawns on the
  *positive*-local-x side of its target gate. That assumption was true for the old fixed spawn by
  construction (fixed negative `x_local`) — does it survive randomizing the gate and adding jitter?
  Yes: `x_local` stays strictly negative (I only randomized its *magnitude*, `1.0–3.0` instead of a
  fixed `2.0`), and the same rotation math applies regardless of which gate's `theta` is used, so the
  drone still ends up on the positive-local-x side of whatever gate got sampled, every time. The
  *sign* is what the sentinel depends on, not the exact spawn distance — worth confirming explicitly
  rather than assuming a change elsewhere didn't quietly invalidate it.
- **Heading**: kept the existing `atan2`-toward-gate formula (already correctly accounts for lateral
  jitter, since it computes the angle from the *actual* jittered spawn point to the gate, not a fixed
  offset) — just widened the yaw noise from the stub's `±0.15` to `±0.3` rad, since there's now real
  heading diversity across gates/positions to cover, not one fixed approach angle.
- **Initial velocity**: new — previously always zero. Direction computed toward the sampled gate
  (normalized spawn→gate vector), magnitude `uniform(0, 3) m/s`. This is a genuinely first-principles
  guess (plausible approach speed given this track's gate spacing), not a derived or tuned number —
  flagging that explicitly since it's exactly the kind of thing to revisit once there's real racing
  data to look at.

**Conclusion / next step.** Implement, syntax-check, then straight to 2e — a real full-track training
run, since 2a-2d are all in place after this.

---
