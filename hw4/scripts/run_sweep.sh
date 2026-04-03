#!/usr/bin/env bash
# format_copy + GRPO ablation sweep
# All jobs are submitted with --detach and run in parallel on Modal.
# Run from the repo root: bash scripts/run_sweep.sh

set -euo pipefail

BASE=(
  --task format_copy
  --algo grpo
  --steps 51
  --batch_size 8
  --group_size 6
  --min_new_tokens 1
  --max_new_tokens 24
  --lr 3e-5
  --ppo_epochs 2
  --minibatch_size 8
  --grad_accum_steps 6
  --clip_eps 0.2
  --kl_coef 0.05
  --max_grad_norm 0.5
  --wandb_enabled
  --wandb_project llm-rl-hw4
  --sample_markdown_log_interval 1
  --sample_log_interval 10
  --sample_log_n 6
  --eval_interval 50
  --save_interval 50
  --warmup_steps 10
)

submit() {
  local name=$1; shift
  echo "Submitting: $name"
  uv run modal run --detach scripts/modal_train.py -- \
    "${BASE[@]}" \
    --output_dir "/vol/runs/${name}" \
    --wandb_name "${name}" \
    "$@"
}

# ── Baseline ──────────────────────────────────────────────────────────────────
submit sweep_baseline

# ── PPO epochs ────────────────────────────────────────────────────────────────
submit sweep_ppo1  --ppo_epochs 1
submit sweep_ppo4  --ppo_epochs 4
submit sweep_ppo6  --ppo_epochs 6

# ── KL coefficient ────────────────────────────────────────────────────────────
submit sweep_kl_0001  --kl_coef 0.001
submit sweep_kl_0005  --kl_coef 0.005
submit sweep_kl_01    --kl_coef 0.1
submit sweep_kl_05    --kl_coef 0.5

# ── Clip epsilon ──────────────────────────────────────────────────────────────
submit sweep_clip_005  --clip_eps 0.05
submit sweep_clip_01   --clip_eps 0.1
submit sweep_clip_03   --clip_eps 0.3
submit sweep_clip_05   --clip_eps 0.5

# ── Grad accum steps (minibatch size held at 8) ───────────────────────────────
submit sweep_accum1   --grad_accum_steps 1
submit sweep_accum3   --grad_accum_steps 3
submit sweep_accum12  --grad_accum_steps 12

# ── Minibatch size (grad_accum_steps held at 6) ───────────────────────────────
submit sweep_mb4   --minibatch_size 4
submit sweep_mb16  --minibatch_size 16
submit sweep_mb24  --minibatch_size 24

echo "All 18 jobs submitted."
