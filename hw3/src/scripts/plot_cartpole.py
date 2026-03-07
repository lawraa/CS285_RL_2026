import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def find_logs(exp_dir: str, pattern: str = "*"):
    return sorted(glob.glob(os.path.join(exp_dir, pattern, "log.csv")))


def plot(log_files: list[str], title: str, output: str):
    _, ax = plt.subplots(figsize=(8, 5))

    for path in log_files:
        df = pd.read_csv(path)
        if "step" not in df.columns or "Eval_AverageReturn" not in df.columns:
            print(f"Skipping {path}: missing required columns")
            continue
        df = df.dropna(subset=["Eval_AverageReturn"])
        label = os.path.basename(os.path.dirname(path))
        ax.plot(df["step"], df["Eval_AverageReturn"], label=label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Average Return")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved plot to {output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp"),
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern to filter experiment subdirectories (e.g. '*CartPole*')",
    )
    parser.add_argument(
        "--log_files",
        type=str,
        nargs="*",
        default=None,
        help="Explicit paths to log.csv files (overrides --exp_dir search)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="DQN Eval Average Return",
        help="Plot title",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_return.png",
        help="Output image file",
    )
    args = parser.parse_args()

    if args.log_files:
        log_files = args.log_files
    else:
        log_files = find_logs(args.exp_dir, args.pattern)

    if not log_files:
        print(f"No log files found under {args.exp_dir} with pattern '{args.pattern}'")
        return

    print(f"Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  {f}")

    plot(log_files, args.title, args.output)


if __name__ == "__main__":
    main()
