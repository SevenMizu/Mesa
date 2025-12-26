"""Execute all 4 policy scenarios from scenarios.yaml and save results.

This script loads the scenario configuration, runs experiments for each
scenario across both US and UK regimes, and saves per-scenario results
to CSV files in a results/ subdirectory.
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from model import HouseholdSavingsModel


def run_experiment_scenario(
    regime: str,
    scenario_name: str,
    scenario_params: dict,
    experiment_params: dict,
) -> pd.DataFrame:
    """Execute a single scenario for one regime and return final-year indicators.

    Parameters
    - regime: str
        "US" or "UK"
    - scenario_name: str
        Name of the scenario for reporting/logging.
    - scenario_params: dict
        Regime-specific parameters (tax_credit_rate, auto_enroll, interest_rate, etc).
    - experiment_params: dict
        Experiment-level settings (runs, years, N, etc).

    Returns
    - pd.DataFrame
        One row per run with final-year indicators.
    """

    results = []
    runs = experiment_params["runs"]
    years = experiment_params["years"]

    for r in range(runs):
        model = HouseholdSavingsModel(
            N=experiment_params["N"],
            regime=regime,
            years=years,
            tax_credit_rate=scenario_params.get("tax_credit_rate", 0.0),
            auto_enroll=scenario_params.get("auto_enroll", False),
            mandatory_contrib_rate=scenario_params.get("mandatory_contrib_rate", 0.0),
            base_savings_rate=scenario_params.get("base_savings_rate", 0.1),
            income_volatility=experiment_params.get("income_volatility", 0.05),
            bias_strength=experiment_params.get("bias_strength", 0.5),
            interest_rate=scenario_params.get("interest_rate", 0.02),
        )

        for _ in range(years):
            model.step()

        df = model.datacollector.get_model_vars_dataframe()
        last = df.iloc[-1]

        results.append({
            "run": r,
            "scenario": scenario_name,
            "regime": regime,
            "Mean_Savings": last.get("Mean_Savings", np.nan),
            "Median_Savings": last.get("Median_Savings", np.nan),
            "Savings_Rate": last.get("Savings_Rate", np.nan),
            "Gini_Wealth": last.get("Gini_Wealth", np.nan),
        })

    print(f"✓ Completed {scenario_name} / {regime} ({runs} runs)")
    return pd.DataFrame(results)


def main():
    """Load scenarios.yaml, execute all scenarios, and save results."""

    # Load scenario configuration
    with open("scenarios.yaml", "r") as f:
        config = yaml.safe_load(f)

    scenarios = config["scenarios"]
    experiment_params = config["experiment"]

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_results = []

    # Run each scenario for both US and UK
    for scenario_name, scenario_config in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"    {scenario_config['description'][:80]}...")

        for regime in ["US", "UK"]:
            regime_params = scenario_config[regime]
            df_scenario = run_experiment_scenario(
                regime=regime,
                scenario_name=scenario_name,
                scenario_params=regime_params,
                experiment_params=experiment_params,
            )
            all_results.append(df_scenario)

    # Combine all results and save
    combined_df = pd.concat(all_results, ignore_index=True)
    output_path = results_dir / "all_scenarios_results.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved combined results to {output_path}")

    # Save per-scenario summaries
    for scenario_name in scenarios.keys():
        scenario_data = combined_df[combined_df["scenario"] == scenario_name]
        summary_path = results_dir / f"scenario_{scenario_name}.csv"
        scenario_data.to_csv(summary_path, index=False)
        print(f"✓ Saved {scenario_name} results to {summary_path}")

    # Print high-level summary
    print("\n=== Summary Statistics ===")
    for scenario_name in scenarios.keys():
        scenario_data = combined_df[combined_df["scenario"] == scenario_name]
        for regime in ["US", "UK"]:
            regime_data = scenario_data[scenario_data["regime"] == regime]
            if not regime_data.empty:
                mean_savings = regime_data["Mean_Savings"].mean()
                gini = regime_data["Gini_Wealth"].mean()
                print(f"{scenario_name:25s} | {regime:2s} | mean savings: {mean_savings:12,.0f} | Gini: {gini:.3f}")


if __name__ == "__main__":
    main()
