import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def find_logs(exp_dir: str, pattern: str = "*"):
    return sorted(glob.glob(os.path.join(exp_dir, pattern, "log.csv")))


def plot(log_files: list[str], output: str):
    _, ax = plt.subplots(figsize=(9, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, path in enumerate(log_files):
        df = pd.read_csv(path)
        color = colors[i % len(colors)]
        run_name = os.path.basename(os.path.dirname(path))

        # Eval return — plot only rows where it is present
        eval_df = df.dropna(subset=["Eval_AverageReturn"])
        if not eval_df.empty:
            ax.plot(
                eval_df["step"],
                eval_df["Eval_AverageReturn"],
                color=color,
                linewidth=2,
                label=f"{run_name} (eval)",
            )

        # Train return — plot only rows where it is present
        train_df = df.dropna(subset=["Train_EpisodeReturn"])
        if not train_df.empty:
            ax.plot(
                train_df["step"],
                train_df["Train_EpisodeReturn"],
                color=color,
                linewidth=0.8,
                alpha=0.5,
                linestyle="--",
                label=f"{run_name} (train)",
            )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Return")
    ax.set_title("DQN Training vs Eval Return")
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
        "--output",
        type=str,
        default="train_vs_eval_return.png",
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

    plot(log_files, args.output)


if __name__ == "__main__":
    main()
