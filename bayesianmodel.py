# ============================================================
# Bayesian Hierarchical Model for Intrinsic Timescales
# subject -> session -> pid -> neuron
# Partial pooling for anatomical region
# Run: python bayesianmodel.py
# ============================================================

import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
from pathlib import Path

# -----------------------------
# 1. Paths
# -----------------------------
timescale_path = "good_isttc.csv"
registry_path  = "ibl_master_pid_registry.csv"

out_dir = Path("bayesian_timescale_outputs")
out_dir.mkdir(exist_ok=True)

# -----------------------------
# 2. Load data
# -----------------------------
df = pd.read_csv(timescale_path)
reg = pd.read_csv(registry_path)

tau_col     = "tau_effective_ms"
pid_col     = "pid"
session_col = "session_id"
region_col  = "beryl_region"
subject_col = "subject"

if session_col not in reg.columns and "eid" in reg.columns:
    reg = reg.rename(columns={"eid": session_col})

reg[subject_col] = (
    reg[subject_col]
    .astype(str)
    .str.replace('KS_091', 'KS091', regex=False)
    .str.strip()
)

reg_keep = reg[[pid_col, session_col, subject_col]].drop_duplicates()

df = df.merge(reg_keep, on=[pid_col, session_col], how="left")

df = df.dropna(subset=[tau_col, region_col, subject_col, session_col, pid_col]).copy()
df = df[df[tau_col] > 0].copy()

df["log_tau"] = np.log10(df[tau_col])

# -----------------------------
# 3. Filtering
# -----------------------------
df[subject_col] = df[subject_col].str.replace('KS_091', 'KS091', regex=False)

df = df[~df[region_col].isin(['root', 'void'])].copy()

min_neurons_per_region = 15
region_counts = df[region_col].value_counts()
valid_regions = region_counts[region_counts >= min_neurons_per_region].index
df = df[df[region_col].isin(valid_regions)].copy()

for c in [subject_col, session_col, pid_col, region_col]:
    df[c] = df[c].astype("category")

print("Neurons :", len(df))
print("Subjects:", df[subject_col].nunique())
print("Sessions:", df[session_col].nunique())
print("PIDs    :", df[pid_col].nunique())
print("Regions :", df[region_col].nunique())

# -----------------------------
# 4. Nesting checks
# -----------------------------
nesting_checks = pd.DataFrame({
    "check": [
        "subjects_per_session_max",
        "sessions_per_pid_max",
        "subjects_per_pid_max",
        "sessions_per_subject_mean",
        "pids_per_session_mean"
    ],
    "value": [
        df.groupby(session_col)[subject_col].nunique().max(),
        df.groupby(pid_col)[session_col].nunique().max(),
        df.groupby(pid_col)[subject_col].nunique().max(),
        df.groupby(subject_col)[session_col].nunique().mean(),
        df.groupby(session_col)[pid_col].nunique().mean()
    ]
})

print(nesting_checks)
nesting_checks.to_csv(out_dir / "nesting_checks.csv", index=False)

# -----------------------------
# 5. Bayesian hierarchical model
# -----------------------------
formula = """
log_tau ~ 1
        + (1 | beryl_region)
        + (1 | subject)
        + (1 | session_id)
        + (1 | pid)
"""

model = bmb.Model(
    formula,
    data=df,
    family="gaussian"
)

idata = model.fit(
    draws=1000,
    tune=1000,
    chains=4,
    cores=4,
    target_accept=0.9,
    random_seed=42
)

# -----------------------------
# 6. Diagnostics
# -----------------------------
summary = az.summary(idata)
print(summary)
summary.to_csv(out_dir / "bayesian_model_summary.csv")

az.plot_trace(idata)

# -----------------------------
# 7. Extract variance components
# -----------------------------
print("Posterior variables:")
print(list(idata.posterior.data_vars))

posterior = idata.posterior
sd_vars   = [v for v in posterior.data_vars if "sigma" in v.lower()]
print("Sigma variables:", sd_vars)

var_rows = []
for v in sd_vars:
    vals     = posterior[v].values.flatten()
    var_vals = vals ** 2
    hdi      = az.hdi(var_vals, hdi_prob=0.94)
    var_rows.append({
        "component":        v,
        "variance_mean":    np.mean(var_vals),
        "variance_hdi_3":   hdi[0],
        "variance_hdi_97":  hdi[1]
    })

variance_df = pd.DataFrame(var_rows)
total_var_mean = variance_df["variance_mean"].sum()
variance_df["variance_share_mean"] = variance_df["variance_mean"] / total_var_mean

print(variance_df)
variance_df.to_csv(out_dir / "bayesian_variance_components.csv", index=False)

# -----------------------------
# 8. Posterior region effects
# -----------------------------
region_vars = [v for v in posterior.data_vars if "beryl_region" in v]
print("Region-related variables:", region_vars)

idata.to_netcdf(out_dir / "bayesian_model_idata.nc")

# -----------------------------
# 9. Posterior predictive check
# -----------------------------
ppc = model.predict(idata, kind="pps", inplace=False)
az.plot_ppc(ppc)

print("Done. Outputs saved to:", out_dir)