import os
import pandas as pd
import matplotlib.pyplot as plt


def load_log(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "log.csv")
    df = pd.read_csv(path)
    required = ["Train_EnvstepsSoFar", "Eval_AverageReturn"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}")
    return df


def main():
    # EDIT these to your actual folders
    default_dir = "exp/InvertedPendulum-v4_pendulum_default"
    best_dir = "exp/InvertedPendulum-v4_pendulum_seed3_b1000_lr0.02_lam0.9_l2_s128_sd3_20260221_064051"

    out_dir = "exp/pendulum_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pendulum_default_vs_best.png")

    df_def = load_log(default_dir)
    df_best = load_log(best_dir)

    plt.figure()
    plt.plot(df_def["Train_EnvstepsSoFar"], df_def["Eval_AverageReturn"], label="default (b=5000)")
    plt.plot(df_best["Train_EnvstepsSoFar"], df_best["Eval_AverageReturn"], label="best (b=1000)")
    plt.axhline(1000, linestyle="--")  # threshold line

    plt.xlabel("Environment steps (Train_EnvstepsSoFar)")
    plt.ylabel("Eval_AverageReturn")
    plt.title("InvertedPendulum-v4: Default vs Tuned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()