import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_curve(run_dir):
    # log.csv is written by Logger(...)
    path = os.path.join(run_dir, "log.csv")
    df = pd.read_csv(path)

    # The logger usually has a step column; handle common variants
    step_col = None
    for c in ["step", "Step", "global_step", "timestep"]:
        if c in df.columns:
            step_col = c
            break
    if step_col is None:
        raise ValueError(f"No step column found in {path}. Columns={list(df.columns)}")

    # Eval metric name used in your run_dqn.py is "Eval_AverageReturn"
    if "Eval_AverageReturn" not in df.columns:
        raise ValueError(f"Eval_AverageReturn not found in {path}. Columns={list(df.columns)}")

    # Keep only rows where eval exists (others may be NaN)
    sub = df[[step_col, "Eval_AverageReturn"]].dropna()
    return sub[step_col].to_numpy(), sub["Eval_AverageReturn"].to_numpy()

# === EDIT THESE: point to your 4 run directories ===
runs = {
    "target_update=500":  "exp/LunarLander-v2_dqn_sd1_tup500_20260304_213659",
    "target_update=1000": "exp/LunarLander-v2_dqn_sd1_tup1000_20260304_185030",
    "target_update=2000": "exp/LunarLander-v2_dqn_sd1_tup2000_20260304_213830",
    "target_update=5000": "exp/LunarLander-v2_dqn_sd1_tup5000_20260304_214019",
    "target_update=10000": "exp/LunarLander-v2_dqn_sd1_tup10000_20260305_061039",
    "target_update=20000": "exp/LunarLander-v2_dqn_sd1_tup20000_20260305_063711",
    "target_update=50000": "exp/LunarLander-v2_dqn_sd1_tup50000_20260305_072355",
}

plt.figure()
for label, run_dir in runs.items():
    x, y = load_curve(run_dir)
    plt.plot(x, y, label=label)

plt.xlabel("Training step")
plt.ylabel("Eval_AverageReturn")
plt.title("LunarLander DQN hyperparameter sensitivity (target update period)")
plt.legend()
plt.tight_layout()
# save 
out_dir = "exp/plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "lunarlander_dqn_target_update_sensitivity.png")
plt.savefig(out_path, dpi=200)
print("Saved plot to:", out_path)