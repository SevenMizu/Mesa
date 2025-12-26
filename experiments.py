import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import HouseholdSavingsModel

"""Experiment utilities for the Household Savings ABM.

This module provides simple helpers to run Monte Carlo experiments
against `HouseholdSavingsModel`, summarize the final-year indicators
across runs, and produce a small plot showing per-run metrics.

Functions
- `run_experiment`: run multiple Monte Carlo runs and collect final-year
    indicators into a DataFrame.
- `summarize`: print basic mean/std statistics for selected metrics.
- `plot_results`: create a small line plot of metrics by run and save it.
"""

def run_experiment(
    regime: str,
    runs: int = 100,
    years: int = 30,
    N: int = 1000,
    **model_kwargs,
) -> pd.DataFrame:
    """Run `runs` Monte Carlo simulations of the ABM and return results.

    Parameters
    - regime: str
        Identifier for the policy/regime settings passed to the model
        (e.g., 'US' or 'UK').
    - runs: int
        Number of independent Monte Carlo runs to perform.
    - years: int
        Number of model steps (time periods) to simulate per run.
    - N: int
        Number of agents/households to instantiate in the model.
    - model_kwargs: dict
        Additional keyword arguments forwarded to `HouseholdSavingsModel`.

    Returns
    - pd.DataFrame
        One row per run containing the final-year indicators collected
        from the model (e.g., mean savings, savings rate, wealth Gini).
    """
    results = []

    for r in range(runs):
        model = HouseholdSavingsModel(
            N=N,
            regime=regime,
            years=years,
            **model_kwargs
        )

        for t in range(years):
            model.step()

        df = model.datacollector.get_model_vars_dataframe()
        last = df.iloc[-1]

        results.append({
            "run": r,
            "regime": regime,
            "Mean_Savings": last.get("Mean_Savings", np.nan),
            "Savings_Rate": last.get("Savings_Rate", np.nan),
            "Gini_Wealth": last.get("Gini_Wealth", np.nan),
        })

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame, label: str) -> None:
    """Print mean and standard deviation for key metrics.

    This helper prints a compact summary for columns `Mean_Savings`,
    `Savings_Rate`, and `Gini_Wealth` when present in `results`.
    """
    print(f"\n=== {label} ===")
    for col in ["Mean_Savings", "Savings_Rate", "Gini_Wealth"]:
        if col in results.columns and not results[col].isna().all():
            mean = results[col].mean()
            std = results[col].std()
            print(f"{col}: mean={mean:.4f}, std={std:.4f}")


def plot_results(results: pd.DataFrame, label: str, output_path: str) -> None:
    """Create and save a simple per-run line plot for available metrics.

    The function looks for the columns `Mean_Savings`, `Savings_Rate`,
    and `Gini_Wealth` in `results`. If none are available or `results`
    is empty, it prints a message and returns early.

    Parameters
    - results: pd.DataFrame
        DataFrame produced by `run_experiment` with one row per run.
    - label: str
        A short label used in the plot title.
    - output_path: str
        Filesystem path where the PNG plot will be saved.
    """
    metrics = [col for col in ["Mean_Savings", "Savings_Rate", "Gini_Wealth"] if col in results.columns]

    if results.empty or not metrics:
        print(f"No data available to plot for {label}.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for metric in metrics:
        ax.plot(results["run"], results[metric], marker="o", label=metric)

    ax.set_title(f"{label} metrics by run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}.")


if __name__ == "__main__":
    # Run baseline experiments for the two regimes and collect results.
    us_baseline = run_experiment(regime="US", runs=100, years=30, N=1000)
    uk_baseline = run_experiment(regime="UK", runs=100, years=30, N=1000)

    # Print compact summaries to stdout.
    summarize(us_baseline, "US baseline")
    summarize(uk_baseline, "UK baseline")

    # Produce small per-run plots and save them as PNG files.
    plot_results(us_baseline, "US baseline", "plot_us_baseline.png")
    plot_results(uk_baseline, "UK baseline", "plot_uk_baseline.png")

    # Persist tabular results for downstream analysis.
    us_baseline.to_csv("results_us_baseline.csv", index=False)
    uk_baseline.to_csv("results_uk_baseline.csv", index=False)

    print("\\nFinished baseline experiments.")

