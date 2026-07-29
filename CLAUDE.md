# ese651_project_redo — CLAUDE.md

## What this repo is

A **solo, from-scratch redo of the simulation half (Phase 1 only)** of the ESE 6510 "Physical
Intelligence" drone racing project. The class competition already happened — Josh and his
teammate Kevin built a working version together (see the sibling `drone-project/` directory for
that full history, results, and a 17-second 3-lap Powerloop time in simulation). This repo is a
**separate, deliberate do-over**: Josh wants to watch an ML engineer work through the same
assignment independently, end to end, with the reasoning made explicit at every step — not to
produce a better result, but so he can learn the *process*.

**This is a from-scratch attempt. Do not copy PPO, reward, observation, or reset code from the
sibling `drone-project/drone-racing` or `drone-project/archive/my_ese651_project` repos.** Reading
them for inspiration on the underlying RL/robotics concepts is fine, but the actual implementation
here should be your own reasoning, tried and iterated on its own merits. If you land on a similar
solution to Kevin's, that's fine — the point is arriving there through your own hypothesis-and-test
process, not copy-pasting.

## Scope: simulation only

Just Phase 1 (train a racing policy in Isaac Lab). **Not** Phase 2 (sim2real deployment on a real
Crazyflie) — there's no more Pennovation access or hardware available now that the season's over,
so that phase isn't reproducible. Stop once you have a trained, evaluated simulation policy and a
written account of how you got there.

## Repo / branch structure

- `main` — mirrors `Jirl-upenn/ese651_project` (the class's canonical template repo) as-is. Left
  untouched as a clean reference.
- `sim-redo` (**default branch, work here**) — branched from commit `1e87843` ("added powerloop
  track and made as default track", 2026-02-10), which is the exact commit Josh and Kevin's own
  `drone-racing` repo was originally rooted from — i.e., this is genuinely "the project as
  received," not the template's current state (the upstream `main` has since moved on to
  `5cf5166`, a bugfix made mid-semester after the class had already started — deliberately not
  used here, since the goal is redoing what was actually handed out).
- `origin` = `Josh0323/ese651_project_redo` (this fork). `upstream` = `Jirl-upenn/ese651_project`
  (the real source, for reference/diffing only — don't push there).

One already-made change on `sim-redo` vs. the pristine commit: `wandb_project` in
`src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py` was changed from
`"ese651_quadcopter"` (the shared project Kevin's `drone-racing` repo logs to, under the `ese651`
W&B team Josh is a member of) to **`"ese651_quadcopter_josh_redo"`**. This is just a project name
string — W&B creates the new project automatically on first `wandb.init()` call, nothing about the
existing `ese651_quadcopter` project is touched, modified, or at any risk. **Do not rename it back,
and do not do anything — ever — that could delete or modify the `ese651_quadcopter` project or
anything else in the `ese651` W&B team.** If you ever think a delete/reset of anything in that
shared team space would help, stop and ask Josh first; there's no scenario in this redo that
should require it.

## Environment — already set up, do not redo

Isaac Sim 4.5, Isaac Lab, and all Python dependencies are **already installed** on the GCP VM used
for the original project. Do not reinstall or reconfigure any of this — it would waste real time
and money for no benefit. See `../drone-project/CLAUDE.md` for full infrastructure details;
summary:

- VM: `instance-20260302-161614`, zone `us-central1-a`, `g2-standard-8` + 1x NVIDIA L4, project
  `project-8525c37b-384c-4bab-ab0`. Conda env `env2` on the VM (`/home/joshkimlab/anaconda3/envs/env2`).
- IsaacLab is already cloned at `/home/joshkimlab/IsaacLab` on the VM. Per the handout, this repo
  needs to be cloned as a **sibling** directory to it — clone this repo (the `sim-redo` branch) to
  something like `/home/joshkimlab/ese651_project_redo` on the VM, not inside `IsaacLab`.
- **A $50/month GCP budget alert already exists**, but that's a safety net, not permission to
  leave the VM running. **Always run
  `gcloud compute instances stop instance-20260302-161614 --zone=us-central1-a` when you're done
  with a session**, and prefer short smoke-test runs (few iterations, small `num_envs`) before
  committing to a long/expensive one. Compute is roughly $0.854/hr — cheap per run, but don't be
  wasteful either.
- `WANDB_API_KEY` is already set in the VM's `~/.bashrc` (under user `joshkimlab`) — non-interactive
  shells (e.g. `sudo -u joshkimlab -H bash -c '...'` over SSH) don't source `.bashrc` automatically,
  so export it explicitly: `export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d= -f2)`
  before running training.
- **`ffmpeg` is installed locally** (on the Mac this session runs from) — use it to inspect any
  video output directly rather than relying on thumbnail workarounds.

## Known infrastructure quirks worth knowing up front

These cost real debugging time today; avoid repeating them:

1. **PYTHONPATH.** Running `python scripts/rsl_rl/train_race.py` directly (not via `python -m`)
   puts the *script's own directory* on `sys.path`, not the repo root, so `import
   src.isaac_quad_sim2real...` fails with `ModuleNotFoundError: No module named 'src'`. Fix:
   `PYTHONPATH=<repo-root>:$PYTHONPATH python scripts/rsl_rl/train_race.py ...`
2. **W&B runs can show "Crashed" on wandb.ai even when training completed successfully.** Isaac
   Sim's `simulation_app.close()` can hard-exit the process, skipping Python's `atexit` handlers
   (including wandb's finish hook). This template's `train_race.py` may or may not already have
   the fix depending on whether it was ported over — check whether `runner.writer.stop()` is
   called after `runner.learn()` and before `env.close()` / `simulation_app.close()`. If missing,
   add it (see `OnPolicyRunner`'s `self.writer` / `WandbSummaryWriter.stop()` in
   `src/third_parties/rsl_rl_local/rsl_rl/utils/wandb_utils.py` — it already defines `stop()`,
   calling `wandb.finish()`, it's just a matter of whether anything invokes it).
3. **`pgrep -f <pattern>` run over `sudo -u joshkimlab -H bash -c '...'` self-matches** — the whole
   script text becomes the process's own argv, so grepping for a substring that appears in your
   own command matches your own check, not the training process. Use `kill -0 <PID>` on a captured
   PID instead when polling for a specific process's liveness.
4. Background SSH watchers (long `until ...; do sleep 30; done` loops via `run_in_background`)
   occasionally die silently without erroring or notifying — verify a run's actual state directly
   (GPU utilization via `nvidia-smi`, log tail, W&B dashboard) rather than fully trusting a single
   watcher for anything time-sensitive.
5. `train_race.py` needs `--video` explicitly passed to record training videos (default off) — it
   slows training down noticeably (periodic video capture overhead), so leave it off for real
   training runs and only enable it for `play_race.py` evaluation clips (or a final training run
   you specifically want on video).

## The actual assignment (transcribed from the course handouts, so you don't need the PDFs)

### Section 2 — PPO
Implement Proximal Policy Optimization by writing the `update()` method marked `#TODO` in
`src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py`. Optionally also explore different
advantage computation in `compute_returns()` in
`src/third_parties/rsl_rl_local/rsl_rl/storage/rollout_storage.py`. This is a local, modified copy
of the `rsl_rl` library — do not copy from the public `rsl_rl` GitHub repo (an academic integrity
issue per the original handout, and also just not the point of the exercise here); reading the
public repo to understand *how* PPO algorithms are typically implemented is fine, writing your own
version from that understanding is the actual task.

Once implemented, a training run should produce a policy that hovers near gate 0 without
triggering the gate-pass condition — that's the expected first checkpoint, not a working racer
yet. Reward curves at that point: `Episode_Reward/progress_goal` climbing, `Episode_Reward/crash`
staying near zero, `Episode_Reward/gate_pass` never firing.

### Section 3 — Strategy (reward, observations, reset)
Complete the `#TODO`s in `get_rewards()`, `get_observations()`, and `reset_idx()` in
`src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py`. The provided
starter code only produces a hover-to-gate-0 policy — it needs real design work, not just filling
blanks.

1. **`get_rewards()`** — reward the drone for racing through gates with minimal lap time:
   - Detect successful gate traversal (the starter's naive distance-threshold check doesn't
     enforce direction or that the drone actually flew through the opening — worth doing better)
   - Reward forward progress through the course
   - Detect and penalize crashes using contact sensor data
   - Multiply reward components by scales defined in `scripts/rsl_rl/train_race.py` (there's a
     `#TODO` block there too, for defining the scales themselves)
2. **`get_observations()`** — give the policy what it needs for navigation/control:
   - Decide the right frame (world / body / gate-relative) for each observation component and be
     deliberate about it — this choice affects whether the policy generalizes across the whole
     track or overfits to specific gate positions
   - Concatenate into a single observation vector
3. **`reset_idx()`** — define episode start states:
   - Initial position/orientation relative to waypoints/gates
   - Consider randomization for policy robustness (see the domain randomization ranges below —
     the eval-time perturbation is real and should shape how wide your training-time
     randomization needs to be)

You're also encouraged to tune hyperparameters in
`src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py`.

### Track: Powerloop
7 gates (0-6). Gates 2/3 form a "powerloop" — a vertical loop maneuver entering both gates from
the same side. Gates 5/6/0 form a "chicane" — a fast, alternating-turn dash through offset gates.
Gate 3 and gate 6 are the *same physical object*, passed in different directions at different
points in the sequence. See `README.md` / the track waypoint definitions in
`src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_env.py` (`'powerloop'` key in the
tracks dict) for exact gate positions/orientations.

### Evaluation-time domain randomization (why robustness matters, not just raw reward)
The original assignment's grading applied dynamics perturbations at evaluation time — even though
there's no live leaderboard to submit to anymore, treating this as a real constraint (not just
optimizing training-time reward) is the more honest version of the exercise:
```
# TWR (thrust-to-weight)
twr in [0.95, 1.05] * nominal
# Aerodynamics
k_aero_xy in [0.5, 2.0] * nominal
k_aero_z  in [0.5, 2.0] * nominal
# PID gains
kp_omega_rp in [0.85, 1.15] * nominal;  ki_omega_rp in [0.85, 1.15] * nominal;  kd_omega_rp in [0.7, 1.3] * nominal
kp_omega_y  in [0.85, 1.15] * nominal;  ki_omega_y  in [0.85, 1.15] * nominal;  kd_omega_y  in [0.7, 1.3] * nominal
```
A policy overfit to nominal dynamics won't hold up under this. Domain-randomize training beyond
just these ranges (Kevin's team trained with wider ranges, e.g. 0.25x-4x on aero drag, specifically
*because* training-time randomization needs more headroom than the eval-time perturbation itself
to be robust to it) — worth treating as a hypothesis to test empirically rather than a fixed rule.

## How to actually work this — the process Josh wants to see

This is the point of the whole exercise, more than the final metrics:

1. **Reason before implementing.** Before writing PPO or reward code, write down (in the log file
   below) what approach you're taking and why — not just what the code does, but the design
   choice behind it. "I'm using a sign-change gate-pass check instead of a distance threshold
   because X" is the kind of statement that belongs in the log *before* you write the code, not
   just as a code comment after.
2. **Validate incrementally, cheaply.** Get PPO minimally working (confirm the hover-to-gate-0
   checkpoint described above) before touching the reward function. Use short smoke tests (small
   `num_envs`, few `max_iterations`) to catch bugs before committing to a full-length, full-cost
   training run — this saved real debugging time today (a `PYTHONPATH` bug and a W&B logging bug
   were both caught this way, cheaply, instead of discovered after an expensive full run).
3. **Read the metrics, don't just glance at the final reward number.** See
   `../drone-project/METRICS_GUIDE.md` for a full walkthrough of what each W&B/console metric
   means and how to use them diagnostically (PPO-health metrics vs. task-performance metrics,
   decomposing an aggregate reward into its components to find out *why* it moved, etc.) — written
   today specifically to close this gap. Read it before your first real training run, not after.
4. **Iterate like a scientist, not by guessing.** Form a specific hypothesis, run an experiment
   that tests it, look at the actual data, then decide the next step — and write all four of those
   down. For a good worked example of this exact process (including things tried and reverted),
   see `../drone-project/drone-racing/WRITEUP.md` (Kevin's own reward-design writeup) and
   `../drone-project/IMPROVEMENT_LOG.md` (today's session's retraining log) for the *format* to
   emulate — not the content to copy.
5. **Keep a running log as you go**, not a retroactive summary at the end. Create
   `EXPERIMENT_LOG.md` in this repo now (before starting PPO) and update it after every
   meaningful experiment: what you tried, why, what happened (with real numbers/metrics, not
   vague impressions), and what you concluded. Include failures and dead ends, not just what
   worked — that's the actual valuable part for someone trying to learn the process.
6. **Be honest about uncertainty.** If a result is ambiguous or you're not sure why something
   happened, say so in the log rather than presenting a tidier story than the data supports.

## Deliverables

By the end: a trained policy that races the Powerloop track well in simulation, `EXPERIMENT_LOG.md`
documenting the full process per the above, and a final results writeup (reward curves, lap
completion, whatever else the metrics guide suggests is worth reporting) comparable in spirit to
`../drone-project/PROJECT_REPORT.md`'s results section — but this repo's own, separate one,
reflecting this redo's own numbers, not the original team's.
