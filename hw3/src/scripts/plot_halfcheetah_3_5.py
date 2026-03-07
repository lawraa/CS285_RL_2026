import os
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exp")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exp", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_DIR = os.path.join(EXP_DIR, "HalfCheetah-v4_sac_sd1_20260305_161039")
AUTOTUNE_DIR = os.path.join(EXP_DIR, "HalfCheetah-v4_sac_autotune_sd1_20260305_164056")


def load_eval(path):
    df = pd.read_csv(os.path.join(path, "log.csv"))
    return df.dropna(subset=["Eval_AverageReturn"])[["step", "Eval_AverageReturn", "temperature"]].reset_index(drop=True)


def main():
    fixed = load_eval(FIXED_DIR)
    autotune = load_eval(AUTOTUNE_DIR)

    # Truncate both to the shorter max step
    max_step = min(fixed["step"].max(), autotune["step"].max())
    fixed = fixed[fixed["step"] <= max_step]
    autotune = autotune[autotune["step"] <= max_step]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Subplot 1: Eval return ---
    ax1.plot(fixed["step"], fixed["Eval_AverageReturn"], label="Fixed temperature")
    ax1.plot(autotune["step"], autotune["Eval_AverageReturn"], label="Auto-tuned temperature")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Eval Average Return")
    ax1.set_title("HalfCheetah-v4 — Eval Return (Section 3.5)")
    ax1.legend(fontsize=9)

    # --- Subplot 2: Temperature (auto-tuned run only) ---
    autotune_temp = autotune.dropna(subset=["temperature"])
    ax2.plot(autotune_temp["step"], autotune_temp["temperature"], color="tab:orange")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Temperature (α)")
    ax2.set_title("HalfCheetah-v4 — Auto-tuned Temperature (Section 3.5)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "halfcheetah_section_3_5.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
