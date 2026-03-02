import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Where CS285 code saves runs
EXP_ROOT = "exp"

# The five experiments you ran (must match --exp_name prefixes above)
LAMBDA_TAGS = [
    ("0.0", "lunar_lander_lambda0"),
    ("0.95", "lunar_lander_lambda0.95"),
    ("0.98", "lunar_lander_lambda0.98"),
    ("0.99", "lunar_lander_lambda0.99"),
    ("1.0", "lunar_lander_lambda1"),
]

def find_latest_run_dir(env_name: str, exp_prefix: str) -> str:
    """
    Runs are saved like:
      exp/LunarLander-v2_<exp_name>_sd1_YYYYMMDD_HHMMSS/
    We pick the most recent matching directory.
    """
    pattern = os.path.join(EXP_ROOT, f"{env_name}_{exp_prefix}_sd*_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No run dirs found for pattern: {pattern}")
    return matches[-1]

def load_log_csv(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing log.csv in {run_dir}")
    df = pd.read_csv(path)
    # Some logs may not have these columns if something broke early
    required = ["Train_EnvstepsSoFar", "Eval_AverageReturn"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}. Found: {list(df.columns)}")
    return df

def main():
    env = "LunarLander-v2"

    plt.figure()
    used = []

    for lam_str, exp_prefix in LAMBDA_TAGS:
        run_dir = find_latest_run_dir(env, exp_prefix)
        df = load_log_csv(run_dir)

        x = df["Train_EnvstepsSoFar"].to_numpy()
        y = df["Eval_AverageReturn"].to_numpy()

        plt.plot(x, y, label=f"λ={lam_str}")
        used.append(run_dir)

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.title("LunarLander-v2: GAE λ sweep")
    plt.legend()

    # Save one combined plot in exp/ as well
    out_path = os.path.join(EXP_ROOT, "lunar_lander_gae_lambda_sweep.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    # Also save a copy into each run folder (nice for submission)
    for run_dir in used:
        per_run_out = os.path.join(run_dir, "gae_lambda_sweep.png")
        try:
            plt.savefig(per_run_out, dpi=200, bbox_inches="tight")
        except Exception as e:
            print(f"Warning: couldn't save into {run_dir}: {e}")

    print(f"Saved combined plot to: {out_path}")
    print("Used runs:")
    for d in used:
        print("  ", d)

if __name__ == "__main__":
    main()