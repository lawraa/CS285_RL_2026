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
submit sweep_kl_2      --kl_coef 2.0
submit sweep_kl_5      --kl_coef 5.0                                                                                           
submit sweep_clip_001  --clip_eps 0.01
submit sweep_kl5_ppo6  --kl_coef 5.0 --ppo_epochs 6  

echo "All 18 jobs submitted."
