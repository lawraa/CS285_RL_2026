import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

EXP_ROOT = "exp"                 # where run.py writes logs
OUT_DIR = os.path.join(EXP_ROOT, "plots")  # save figures here

# Map "experiment label" -> substring to find in folder name
# run.py creates folders like: exp/CartPole-v0_<exp_name>_sd1_<timestamp>
SMALL = {
    "traj": "CartPole-v0_cartpole_",
    "rtg": "CartPole-v0_cartpole_rtg_",
    "traj+na": "CartPole-v0_cartpole_na_",
    "rtg+na": "CartPole-v0_cartpole_rtg_na_",
}

LARGE = {
    "traj_lb": "CartPole-v0_cartpole_lb_",
    "rtg_lb": "CartPole-v0_cartpole_lb_rtg_",
    "traj_lb+na": "CartPole-v0_cartpole_lb_na_",
    "rtg_lb+na": "CartPole-v0_cartpole_lb_rtg_na_",
}

def find_latest_run_folder(pattern_substr: str) -> str:
    candidates = [d for d in glob.glob(os.path.join(EXP_ROOT, "*")) if os.path.isdir(d)]
    matches = [d for d in candidates if pattern_substr in os.path.basename(d)]
    if not matches:
        raise FileNotFoundError(f"No run folder found containing substring: {pattern_substr}")
    # pick latest by timestamp in folder name (lexicographic works with YYYYMMDD_HHMMSS)
    matches.sort()
    return matches[-1]

def load_log(folder: str) -> pd.DataFrame:
    path = os.path.join(folder, "log.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing log.csv in {folder}")
    df = pd.read_csv(path)
    # Ensure numeric
    for col in ["Train_EnvstepsSoFar", "Train_AverageReturn", "Eval_AverageReturn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def plot_group(group_name: str, mapping: dict, out_png: str, use_eval: bool = False):
    plt.figure()
    for label, substr in mapping.items():
        folder = find_latest_run_folder(substr)
        df = load_log(folder)
        ycol = "Eval_AverageReturn" if use_eval else "Train_AverageReturn"
        plt.plot(df["Train_EnvstepsSoFar"], df[ycol], label=label)
        print(f"{group_name}: {label} -> {os.path.basename(folder)}")

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn" if use_eval else "Train_AverageReturn")
    plt.title(group_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, out_png), dpi=200, bbox_inches="tight")
    plt.close()

def make_summary_csv(out_csv: str):
    rows = []
    all_maps = {**SMALL, **LARGE}
    for label, substr in all_maps.items():
        folder = find_latest_run_folder(substr)
        df = load_log(folder)

        # final values (last logged point)
        last = df.iloc[-1]
        rows.append({
            "label": label,
            "folder": os.path.basename(folder),
            "final_envsteps": int(last["Train_EnvstepsSoFar"]),
            "final_train_return": float(last["Train_AverageReturn"]),
            "final_eval_return": float(last["Eval_AverageReturn"]),
            "max_train_return": float(df["Train_AverageReturn"].max()),
            "max_eval_return": float(df["Eval_AverageReturn"].max()),
        })

    out = pd.DataFrame(rows).sort_values("label")
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, out_csv), index=False)

if __name__ == "__main__":
    # Train curves (deliverable says average return vs env steps; typically use Train_AverageReturn)
    plot_group("CartPole Small Batch (b=1000)", SMALL, "cartpole_small_train.png", use_eval=False)
    plot_group("CartPole Large Batch (b=4000)", LARGE, "cartpole_large_train.png", use_eval=False)

    # Optional: eval curves too (often nicer)
    plot_group("CartPole Small Batch (Eval)", SMALL, "cartpole_small_eval.png", use_eval=True)
    plot_group("CartPole Large Batch (Eval)", LARGE, "cartpole_large_eval.png", use_eval=True)

    # Summary table
    make_summary_csv("summary.csv")

    print(f"\nSaved plots + summary to: {OUT_DIR}")