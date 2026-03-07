import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exp")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exp", "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def find_hopper_logs():
    return sorted(glob.glob(os.path.join(EXP_DIR, "*Hopper*", "log.csv")))


def short_label(path: str) -> str:
    name = os.path.basename(os.path.dirname(path))
    # Extract the meaningful part between env name and seed/timestamp
    # e.g. Hopper-v4_sac_clipq_sd1_... -> sac_clipq
    parts = name.split("_")
    try:
        # Drop the env part (everything up to first part containing '-v')
        env_end = next(i for i, p in enumerate(parts) if "-v" in p)
        # Drop seed (sdN) and timestamp (date_time) from the end
        config_parts = [p for p in parts[env_end + 1:] if not p.startswith("sd") and not p.isdigit() and len(p) != 8]
        return "_".join(config_parts) if config_parts else name
    except StopIteration:
        return name


def plot_metric(log_files, col, ylabel, title, out_file):
    _, ax = plt.subplots(figsize=(8, 5))
    for path in log_files:
        df = pd.read_csv(path).dropna(subset=[col])
        if col not in df.columns or "step" not in df.columns:
            print(f"Skipping {path}: missing column '{col}'")
            continue
        ax.plot(df["step"], df[col], label=short_label(path))
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")
    plt.close()


def main():
    log_files = find_hopper_logs()
    if not log_files:
        print(f"No Hopper logs found under {EXP_DIR}")
        return
    print(f"Found {len(log_files)} Hopper log(s):")
    for f in log_files:
        print(f"  {f}")

    plot_metric(
        log_files,
        col="Eval_AverageReturn",
        ylabel="Eval Average Return",
        title="Hopper-v4 — Eval Return (Section 3.6)",
        out_file=os.path.join(OUT_DIR, "hopper_eval_return.png"),
    )

    plot_metric(
        log_files,
        col="q_values",
        ylabel="Q Values",
        title="Hopper-v4 — Q Values (Section 3.6)",
        out_file=os.path.join(OUT_DIR, "hopper_q_values.png"),
    )


if __name__ == "__main__":
    main()
