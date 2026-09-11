from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def load_history(path):
    """Load an annealing_history_*.csv. One row per temperature step:
    temperature_step, temperature, current_energy, best_energy,
    accepted, acceptance_rate, new_bests, blackbox_calls."""
    return pd.read_csv(path)


def plot_annealing_diagnostics(history, title="", save_path="annealing_diagnostics.png"):
    """Three stacked panels sharing the x-axis (blackbox_calls, since
    that's the real 'cost' axis -- each call is an expensive dispatch-down
    simulation, not a free sweep):
      1. current_energy vs best_energy, with markers where new_bests fired
      2. temperature schedule (log scale)
      3. acceptance_rate over the run
    """
    x = history["blackbox_calls"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 1]})

    ax = axes[0]
    ax.plot(x, history["current_energy"], color="steelblue", alpha=0.5,
            linewidth=1, label="current energy")
    ax.plot(x, history["best_energy"], color="crimson", linewidth=2,
            label="best-so-far energy")
    improved = history[history["new_bests"] > 0]
    if len(improved):
        ax.scatter(improved["blackbox_calls"], improved["best_energy"],
                    color="crimson", zorder=5, s=25, label="new best found")
    ax.set_ylabel("dispatch down %")
    ax.set_title(title or "Annealing run diagnostics")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, history["temperature"], color="darkorange")
    ax.set_yscale("log")
    ax.set_ylabel("temperature (log)")

    ax = axes[2]
    ax.plot(x, history["acceptance_rate"], color="seagreen")
    ax.set_ylabel("acceptance rate")
    ax.set_xlabel("blackbox calls (simulator evaluations)")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def summarize_run(history):
    """Quick text summary: total calls, starting/final energy, total
    improvement, and where in the run most of the gains happened."""
    start_E = history["current_energy"].iloc[0]
    best_E = history["best_energy"].iloc[-1]
    total_calls = history["blackbox_calls"].iloc[-1]
    n_new_bests = history["new_bests"].sum()
    improvement = start_E - best_E
    print(f"Total blackbox calls:   {total_calls}")
    print(f"Starting energy:        {start_E:.4f}")
    print(f"Final best energy:      {best_E:.4f}")
    print(f"Improvement:            {improvement:.4f} ({100*improvement/start_E:.1f}% relative)")
    print(f"New-best events:        {int(n_new_bests)}")
    return dict(total_calls=total_calls, start_E=start_E, best_E=best_E,
                improvement=improvement, n_new_bests=int(n_new_bests))


if __name__ == "__main__":
    history = load_history(OUTPUT_DIR / "annealing_history_long.csv")
    summarize_run(history)
    plot_annealing_diagnostics(history, title="6 counties, 0.95 alpha, 20 sweeps per temperature",
                                save_path=OUTPUT_DIR / "hist/annealing_diagnostics_6_counties.png")
    print("done")