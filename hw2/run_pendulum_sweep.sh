#!/bin/bash

# Activate uv environment automatically if needed
set -e

ENV="InvertedPendulum-v4"

echo "===== STARTING PENDULUM SWEEP ====="

# Seeds (very important in RL)
SEEDS=(1 2 3 4 5)

# Batch sizes
BATCHES=(1000 1500 2000 3000)

# Learning rates
LRS=(0.02 0.01 0.005)

# GAE lambda values
LAMBDAS=(0.9 0.95 0.98)

# Network sizes
LAYERS=(2 3)
SIZES=(32 64 128)

for seed in "${SEEDS[@]}"; do
  for b in "${BATCHES[@]}"; do
    for lr in "${LRS[@]}"; do
      for lam in "${LAMBDAS[@]}"; do
        for l in "${LAYERS[@]}"; do
          for s in "${SIZES[@]}"; do

            EXP="pendulum_seed${seed}_b${b}_lr${lr}_lam${lam}_l${l}_s${s}"

            echo "Running $EXP"

            uv run src/scripts/run.py \
              --env_name $ENV \
              -n 100 \
              -b $b \
              -eb 1000 \
              --discount 0.99 \
              -lr $lr \
              -l $l \
              -s $s \
              --use_reward_to_go \
              --use_baseline \
              --normalize_advantages \
              --gae_lambda $lam \
              -bgs 5 \
              -blr 0.01 \
              --seed $seed \
              --exp_name $EXP

          done
        done
      done
    done
  done
done

echo "===== SWEEP FINISHED ====="