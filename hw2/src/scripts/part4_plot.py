import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def find_runs(exp_root: str, env_prefix: str, exp_names: list[str]):
    """
    Finds exp folders like:
      exp/HalfCheetah-v4_cheetah_sd1_YYYYMMDD_HHMMSS/log.csv
    """
    runs = []
    pattern = os.path.join(exp_root, f"{env_prefix}_*_sd*_*")
    for d in sorted(glob.glob(pattern)):
        flags_path = os.path.join(d, "flags.json")
        log_path = os.path.join(d, "log.csv")
        if not (os.path.exists(flags_path) and os.path.exists(log_path)):
            continue

        try:
            flags = pd.read_json(flags_path, typ="series")
        except Exception:
            continue

        name = str(flags.get("exp_name", ""))
        if name in exp_names:
            runs.append((name, d, log_path, flags_path))
    return runs


def load_log(log_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(log_csv_path)
    # expected columns include:
    # Train_EnvstepsSoFar, Eval_AverageReturn, Baseline Loss (only for baseline run)
    return df


def plot_eval_return(runs, out_path: str, title: str):
    plt.figure()
    for name, d, log_path, _ in runs:
        df = load_log(log_path)
        if "Train_EnvstepsSoFar" not in df.columns or "Eval_AverageReturn" not in df.columns:
            print(f"[WARN] Missing columns in {log_path}")
            continue
        plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=name)

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_baseline_loss(runs, out_path: str, title: str):
    plt.figure()
    plotted_any = False
    for name, d, log_path, _ in runs:
        df = load_log(log_path)
        if "Train_EnvstepsSoFar" not in df.columns:
            continue

        # Your critic logger key is "Baseline Loss" (from ValueCritic.update)
        if "Baseline Loss" not in df.columns:
            print(f"[WARN] No 'Baseline Loss' column in {log_path} (run={name}).")
            continue

        plt.plot(df["Train_EnvstepsSoFar"], df["Baseline Loss"], label=name)
        plotted_any = True

    if not plotted_any:
        print("[WARN] No baseline-loss curves plotted (did you run a baseline experiment?).")

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Baseline Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, default="exp", help="Where runs are saved (default: exp/)")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--out_dir", type=str, default="exp", help="Where to save plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # These names match the homework commands.
    # If you changed exp_name, update this list.
    target_names = ["cheetah", "cheetah_baseline"]

    runs = find_runs(args.exp_root, args.env, target_names)
    if len(runs) == 0:
        print("No runs found. Make sure you have exp/*/flags.json and exp/*/log.csv.")
        return

    # Group by exp_name
    by_name = {}
    for name, d, log_path, flags_path in runs:
        by_name.setdefault(name, []).append((name, d, log_path, flags_path))

    # Plot eval return for both (if both exist)
    combined = []
    for nm in target_names:
        combined.extend(by_name.get(nm, []))
    plot_eval_return(
        combined,
        out_path=os.path.join(args.out_dir, f"{args.env}_part4_eval_return.png"),
        title=f"{args.env}: Eval Average Return",
    )

    # Plot baseline loss (only baseline runs will show)
    baseline_runs = by_name.get("cheetah_baseline", [])
    plot_baseline_loss(
        baseline_runs,
        out_path=os.path.join(args.out_dir, f"{args.env}_part4_baseline_loss.png"),
        title=f"{args.env}: Baseline Loss (Value Function)",
    )


if __name__ == "__main__":
    main()