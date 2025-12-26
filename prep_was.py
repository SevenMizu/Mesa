import pandas as pd

# 1. Load WAS Round 8 household data
WAS_FILENAME = "was_round_8_hhold_eul_may_2025_230525.dta"

# convert_categoricals=False avoids issues with duplicate value labels
was_raw = pd.read_stata(WAS_FILENAME, convert_categoricals=False)

print("Columns (first 30):")
print(list(was_raw.columns[:30]))

# 2. Pick variables (update these names after inspecting the columns)

INCOME_VAR_NAME = "DVTotInc_BHCR8"
WEALTH_VAR_NAME = "TotalWlthR8"

was_clean = was_raw[[INCOME_VAR_NAME, WEALTH_VAR_NAME]].copy()

# 3. Rename to the names your Mesa model expects
was_clean = was_clean.rename(columns={
    INCOME_VAR_NAME: "income",
    WEALTH_VAR_NAME: "net_wealth",
})

# 4. Basic cleaning
was_clean = was_clean.dropna(subset=["income", "net_wealth"])
was_clean = was_clean[was_clean["income"] > 0]

# 5. Save as was_clean.csv in the same folder
was_clean.to_csv("was_clean.csv", index=False)

print("Done. Saved was_clean.csv with", len(was_clean), "rows.")
