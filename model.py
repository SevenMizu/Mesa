import random
import numpy as np
import pandas as pd

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

"""
    Load preprocessed household data for a given regime.
    regime: "US" or "UK"
    Returns a pandas DataFrame with at least columns:
        income, net_wealth
    """
def load_household_data(regime: str) -> pd.DataFrame:
    
    if regime == "US":
        path = "scf_clean.csv"
    elif regime == "UK":
        path = "was_clean.csv"
    else:
        raise ValueError(f"Unknown regime: {regime}")

    df = pd.read_csv(path)

    # Basic cleaning: drop missing, keep only positive income
    df = df.dropna(subset=["income", "net_wealth"])
    df = df[df["income"] > 0]

    return df


def sample_initial_households(regime: str, N: int) -> pd.DataFrame:
    """
    Sample N households from SCF or WAS microdata.
    Returns a DataFrame of length N with columns:
        income, net_wealth, savings
    """
    df = load_household_data(regime)

    # Simple random sampling with replacement.
    sampled = df.sample(n=N, replace=True, random_state=None).reset_index(drop=True)

    # Assume some fraction of net_wealth is "savings" (clipped at 0).
    sampled["savings"] = sampled["net_wealth"].clip(lower=0)

    return sampled


def gini_coefficient(x: np.ndarray) -> float:
    """
    Compute Gini coefficient for a 1D numpy array.
    Uses the standard formula based on sorted values.
    """
    x = x.astype(float)
    x = x[x >= 0]  # ignore negative values for simplicity
    if len(x) == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0

    sorted_x = np.sort(x)
    n = len(sorted_x)
    gini = (2.0 * np.sum((np.arange(1, n + 1) * sorted_x)) / (n * np.sum(sorted_x))) - (n + 1) / n
    return gini


class HouseholdAgent(Agent):
    def __init__(
        self,
        unique_id,
        model,
        income: float,
        initial_savings: float,
        consumption_rate: float,
        present_bias: float,
    ):
        super().__init__(unique_id, model)
        self.income = income
        self.savings = initial_savings
        self.consumption_rate = consumption_rate
        self.present_bias = present_bias

        self.pension_contrib = 0.0
        self.tax_credit = 0.0

    def step(self):
        """
        One period update:
        1. Income may be shocked.
        2. Policy is applied (tax credits, auto-enrollment).
        3. Decide consumption and update savings.
        """

        # 1. Income shock (simple idiosyncratic multiplicative noise)
        shock = np.random.normal(loc=1.0, scale=self.model.income_volatility)
        eff_income = max(self.income * shock, 0.0)

        # 2. Apply policy depending on regime and parameters
        policy_bonus = 0.0

        if self.model.regime == "US":
            # simple tax credit for contributions
            if self.model.tax_credit_rate > 0:
                target_savings = self.model.base_savings_rate * eff_income
                tax_credit = self.model.tax_credit_rate * target_savings
                policy_bonus += tax_credit
        elif self.model.regime == "UK":
            # auto enrollment with mandatory contribution rate if enabled
            if self.model.auto_enroll:
                mandatory_contrib = self.model.mandatory_contrib_rate * eff_income
                self.pension_contrib = mandatory_contrib
                policy_bonus += mandatory_contrib

        # 3. Decide consumption
        # simple rule: consume = consumption_rate * effective income adjusted by present bias
        effective_for_decision = eff_income * (
            1.0 + (self.present_bias - 1.0) * self.model.bias_strength
        )
        consumption = self.consumption_rate * effective_for_decision

        # Apply interest on existing savings and update balance.
        interest_earned = self.savings * self.model.interest_rate
        income_plus_policy = eff_income + policy_bonus + interest_earned
        new_savings = self.savings + income_plus_policy - consumption
        self.savings = max(new_savings, 0.0)


class HouseholdSavingsModel(Model):
    def __init__(
        self,
        N: int = 1000,
        regime: str = "US",
        years: int = 30,
        tax_credit_rate: float = 0.15,
        auto_enroll: bool = True,
        mandatory_contrib_rate: float = 0.05,
        base_savings_rate: float = 0.1,
        income_volatility: float = 0.05,
        bias_strength: float = 0.5,
        interest_rate: float = 0.02,
        random_seed: int | None = None,
    ):
        """
        N: number of agents
        regime: "US" or "UK"
        years: time horizon
        tax_credit_rate: used mainly in US regime
        auto_enroll, mandatory_contrib_rate: used mainly in UK regime
        base_savings_rate: target share of income for savings in policy calculation
        income_volatility: std dev of income shocks
        bias_strength: how strongly present_bias affects decision-making
        """
        super().__init__()
        if random_seed is not None:
            self.random.seed(random_seed)
            np.random.seed(random_seed)

        self.N = N
        self.regime = regime
        self.years = years

        self.tax_credit_rate = tax_credit_rate
        self.auto_enroll = auto_enroll
        self.mandatory_contrib_rate = mandatory_contrib_rate
        self.base_savings_rate = base_savings_rate
        self.income_volatility = income_volatility
        self.bias_strength = bias_strength
        self.interest_rate = interest_rate

        self.schedule = RandomActivation(self)

        # load and sample initial agents from SCF or WAS
        sampled = sample_initial_households(regime, N)

        # create agents
        for i in range(N):
            income = sampled.loc[i, "income"]
            initial_savings = sampled.loc[i, "savings"]

            # draw behavioral parameters
            consumption_rate = np.clip(
                np.random.normal(loc=0.8, scale=0.05), 0.6, 0.95
            )
            present_bias = np.clip(
                np.random.normal(loc=0.9, scale=0.05), 0.7, 1.0
            )

            agent = HouseholdAgent(
                unique_id=i,
                model=self,
                income=income,
                initial_savings=initial_savings,
                consumption_rate=consumption_rate,
                present_bias=present_bias,
            )
            self.schedule.add(agent)

        # data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Mean_Savings": self.mean_savings,
                "Median_Savings": self.median_savings,
                "Savings_Rate": self.savings_rate,
                "Gini_Wealth": self.gini_wealth,
            }
        )

    def mean_savings(self):
        savings = np.array([a.savings for a in self.schedule.agents])
        return float(np.mean(savings))
    
    def median_savings(self):
        savings = np.array([a.savings for a in self.schedule.agents])
        return float(np.median(savings))

    def savings_rate(self):
        savings = np.array([a.savings for a in self.schedule.agents])
        incomes = np.array([a.income for a in self.schedule.agents])
        total_inc = np.sum(incomes)
        if total_inc <= 0:
            return np.nan
        return float(np.sum(savings) / total_inc)

    def gini_wealth(self):
        savings = np.array([a.savings for a in self.schedule.agents])
        return float(gini_coefficient(savings))

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
