import pandas as pd

# 1. Load WAS Round 8 household data
WAS_FILENAME = "was_round_8_hhold_eul_may_2025_230525.dta"

print(f"Loading {WAS_FILENAME} ...")
was_raw = pd.read_stata(WAS_FILENAME, convert_categoricals=False)

print("Shape:", was_raw.shape)
print("First 40 columns:")
print(list(was_raw.columns[:40]))


INCOME_VAR_NAME = "DVTotInc_BHCR8"   # TODO: change if this doesn't exist
WEALTH_VAR_NAME = "TotalWlthR8"      # TODO: change if this doesn't exist

if INCOME_VAR_NAME not in was_raw.columns or WEALTH_VAR_NAME not in was_raw.columns:
    print("\n[!] WARNING: Placeholder variable names not found in columns above.")
    print("    Once you see the printed column names, we will update")
    print("    INCOME_VAR_NAME and WEALTH_VAR_NAME to match your file.")
else:
    was_clean = was_raw[[INCOME_VAR_NAME, WEALTH_VAR_NAME]].copy()

    was_clean = was_clean.rename(columns={
        INCOME_VAR_NAME: "income",
        WEALTH_VAR_NAME: "net_wealth",
    })

    was_clean = was_clean.dropna(subset=["income", "net_wealth"])
    was_clean = was_clean[was_clean["income"] > 0]

    was_clean.to_csv("was_clean.csv", index=False)

    print("\nDone. Saved was_clean.csv with", len(was_clean), "rows.")
