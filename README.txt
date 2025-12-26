Household Savings ABM (Mesa, SCF/WAS-based)

Files:
- model.py           : Mesa HouseholdSavingsModel using US (SCF) and UK (WAS) microdata.
- experiments.py     : Monte Carlo experiment runner (US vs UK, baseline).

You must provide two CSV files in the same directory:
- scf_clean.csv   : cleaned US microdata with at least columns "income", "net_wealth"
- was_clean.csv   : cleaned UK microdata with at least columns "income", "net_wealth"

Then install dependencies:
 pip install mesa pandas numpy
