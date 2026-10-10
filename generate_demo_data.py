"""Create a clearly synthetic, reproducible dataset for demonstrating the pipeline."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 1600
age = rng.integers(21, 70, n)
income = np.clip(rng.lognormal(np.log(650000), 0.55, n), 120000, 3500000).astype(int)
dti = np.clip(rng.beta(2.0, 5.0, n), 0.01, 0.95)
util = np.clip(rng.beta(2.2, 3.0, n), 0.01, 1.0)
late = rng.poisson(0.8, n).clip(0, 8)
history = rng.integers(0, 35, n)
employment = rng.choice(["salaried", "self_employed", "contract", "unemployed"], n, p=[.55,.22,.13,.10])
home = rng.choice(["rent", "own", "mortgage", "family"], n, p=[.35,.2,.3,.15])
# Simulated target only; it is not a real-world credit policy.
score = (1.0*(income > 600000) + 0.65*(history > 5) + 0.4*(age > 25)
         - 2.2*dti - 1.1*util - 0.55*late
         + 0.25*(employment == "salaried") + 0.25*(home == "own")
         + rng.normal(0, .8, n))
p = 1/(1+np.exp(-score))
y = (rng.random(n) < p).astype(int)
df = pd.DataFrame({"age":age,"annual_income":income,"debt_to_income":dti.round(3),
 "credit_utilization":util.round(3),"late_payments_12m":late,"credit_history_years":history,
 "employment_type":employment,"home_ownership":home,"creditworthy":y})
os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_demo.csv", index=False)
print(f"Created data/credit_demo.csv with {len(df)} synthetic rows.")
print(df.head().to_string(index=False))
