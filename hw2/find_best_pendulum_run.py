# save as: src/scripts/find_best_pendulum_run.py
#
# Usage:
#   uv run python src/scripts/find_best_pendulum_run.py --exp_root exp --env_prefix InvertedPendulum-v4 --threshold 1000
#
# It scans exp/*/log.csv + flags.json, finds the earliest Train_EnvstepsSoFar where Eval_AverageReturn >= threshold,
# then sorts runs by that earliest step (smaller is better).

import os
import glob
import json
import argparse
import pandas as pd


def safe_load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def find_runs(exp_root: str, env_prefix: str):
    # folders look like: exp/InvertedPendulum-v4_<exp_name>_sd1_YYYYMMDD_HHMMSS/
    pattern = os.path.join(exp_root, f"{env_prefix}_*_sd*_*")
    return sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])


def earliest_step_reaching_threshold(df: pd.DataFrame, threshold: float):
    if "Train_EnvstepsSoFar" not in df.columns or "Eval_AverageReturn" not in df.columns:
        return None
    hit = df[df["Eval_AverageReturn"] >= threshold]
    if hit.empty:
        return None
    # earliest point it reached threshold
    return int(hit["Train_EnvstepsSoFar"].iloc[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, default="exp")
    parser.add_argument("--env_prefix", type=str, default="InvertedPendulum-v4")
    parser.add_argument("--threshold", type=float, default=1000.0)
    parser.add_argument("--only_prefix", type=str, default="pendulum")  # exp_name must start with this
    args = parser.parse_args()

    run_dirs = find_runs(args.exp_root, args.env_prefix)

    results = []
    for d in run_dirs:
        flags_path = os.path.join(d, "flags.json")
        log_path = os.path.join(d, "log.csv")
        if not (os.path.exists(flags_path) and os.path.exists(log_path)):
            continue

        flags = safe_load_json(flags_path)
        if not flags:
            continue

        exp_name = str(flags.get("exp_name", ""))
        if args.only_prefix and not exp_name.startswith(args.only_prefix):
            continue

        try:
            df = pd.read_csv(log_path)
        except Exception:
            continue

        earliest = earliest_step_reaching_threshold(df, args.threshold)
        max_eval = float(df["Eval_AverageReturn"].max()) if "Eval_AverageReturn" in df.columns else float("nan")

        results.append(
            {
                "exp_name": exp_name,
                "run_dir": d,
                "earliest_steps_to_threshold": earliest,  # None if never reached
                "max_eval_return": max_eval,
                "batch_size": flags.get("batch_size"),
                "learning_rate": flags.get("learning_rate"),
                "gae_lambda": flags.get("gae_lambda"),
                "use_baseline": flags.get("use_baseline"),
                "use_reward_to_go": flags.get("use_reward_to_go"),
                "normalize_advantages": flags.get("normalize_advantages"),
                "seed": flags.get("seed"),
            }
        )

    # Sort: (1) runs that reached threshold first, by earliest step asc
    #       (2) then by max return desc for runs that never reached threshold
    reached = [r for r in results if r["earliest_steps_to_threshold"] is not None]
    not_reached = [r for r in results if r["earliest_steps_to_threshold"] is None]

    reached.sort(key=lambda r: r["earliest_steps_to_threshold"])
    not_reached.sort(key=lambda r: -r["max_eval_return"])

    print("\n=== TOP RUNS THAT REACHED THRESHOLD ===")
    for r in reached:
        print(
            f"{r['earliest_steps_to_threshold']:>8} steps | max={r['max_eval_return']:.1f} | "
            f"{r['exp_name']} | seed={r['seed']} | b={r['batch_size']} lr={r['learning_rate']} lam={r['gae_lambda']} "
            f"na={r['normalize_advantages']} baseline={r['use_baseline']} rtg={r['use_reward_to_go']}\n"
            f"    dir: {r['run_dir']}"
        )

    if reached:
        best = reached[0]
        print("\n=== BEST RUN ===")
        print(
            f"BEST: {best['exp_name']}\n"
            f"  earliest Train_EnvstepsSoFar reaching {args.threshold}: {best['earliest_steps_to_threshold']}\n"
            f"  max Eval_AverageReturn: {best['max_eval_return']:.1f}\n"
            f"  run dir: {best['run_dir']}"
        )

    print("\n=== TOP RUNS THAT DID NOT REACH THRESHOLD (sorted by max return) ===")
    for r in not_reached[:10]:
        print(
            f"max={r['max_eval_return']:.1f} | {r['exp_name']} | seed={r['seed']} | b={r['batch_size']} lr={r['learning_rate']} lam={r['gae_lambda']}\n"
            f"    dir: {r['run_dir']}"
        )

    if not results:
        print("No matching runs found. Check exp_root/env_prefix/only_prefix.")
    print(len(reached))

if __name__ == "__main__":
    main()