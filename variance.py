
"""
Estimator x Model x Scale disentangling analysis (revised)
==========================================================
Reviewer's request: separate the sources of the discrepancy between our
24.1% (region variance share) and Shi et al.'s 2.5% (region R^2).

Three factors, 2 x 2 x 2 = 8 cells:

    ESTIMATOR : binned ACF        vs  iSTTC
    MODEL     : ordinary R^2      vs  hierarchical variance share
    SCALE     : raw tau           vs  log10 tau

Fixes relative to the previous version
--------------------------------------
1. Units: every lag vector is forced to SECONDS right after it is computed,
   and every output is called tau_*_ms (explicit units).
2. Passive epoch: the epoch source returned by get_spontaneous_window is
   stored and, by default, anything that is not passive is skipped.
3. IBL QC: good cluster == label 1 (all QC metrics passed). No silent
   try/except: a missing label column raises an error.
4. Same fitting timescale: both estimators are fit on the IDENTICAL lag grid
   (600 lags x 10 ms = 6 s). The binned ACF is computed at 5 ms bins and read
   out on the iSTTC lag grid.
5. Same neurons: ONE common neuron set is built once (both taus valid, > 0,
   metadata complete, regions >= MIN_NEURONS_PER_REGION in that set) and used
   for all 8 cells. n is asserted to be identical across cells.
6. Variance share: computed per posterior draw, then averaged.
7. Nothing runs at import time.

Requirements:
    pip install ONE-api iblatlas pandas numpy scipy numba joblib tqdm bambi arviz scikit-learn

Needs `calculate_isttc`, `fit_multi_exponential`, `get_spontaneous_window`
from your isttc_pipeline.py (same folder).

Usage:
    python dual_estimator_analysis.py            # load IBL data + analysis
    python dual_estimator_analysis.py --reuse    # reuse dual_estimator_tau.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

from isttc_pipeline import (
    calculate_isttc,
    fit_multi_exponential,
    get_spontaneous_window,
)

# =============================================================================
# GLOBAL SETTINGS (one place, so both estimators always share them)
# =============================================================================

LAG_STEP_S = 0.010          # lag spacing of the fit grid (iSTTC lag_shift)
N_LAGS = 600                # -> fit window = 600 x 10 ms = 6 s for BOTH estimators
ISTTC_DELTA_T_S = 0.005     # iSTTC synchronicity window
ACF_BIN_S = 0.005           # binned-ACF bin size (lag grid is a multiple of it)
MIN_SPIKES = 100
MIN_R2_FIT = 0.5
MIN_NEURONS_PER_REGION = 15
IBL_GOOD_LABEL = 1          # IBL standard: label == 1 -> all QC metrics passed
REQUIRE_PASSIVE = True

CSV_PATH = "dual_estimator_tau.csv"
REGISTRY_CSV = "ibl_master_pid_registry.csv"


# =============================================================================
# PART 1: BINNED ACF (Shi et al. 2025, Eq. 1)
# =============================================================================

@njit(cache=True)
def _binned_acf_nb(counts, nlags):
    n = len(counts)
    mean_all = 0.0
    for i in range(n):
        mean_all += counts[i]
    mean_all /= n

    var = 0.0
    for i in range(n):
        d = counts[i] - mean_all
        var += d * d
    var /= (n - 1)

    ac = np.full(nlags, np.nan)
    if var == 0.0:
        return ac

    for j in range(nlags):
        if j >= n:
            break
        m = n - j
        if m < 2:
            break
        mu1 = 0.0
        for i in range(m):
            mu1 += counts[i]
        mu1 /= m
        mu2 = 0.0
        for i in range(j, n):
            mu2 += counts[i]
        mu2 /= m

        s = 0.0
        for i in range(m):
            s += (counts[i] - mu1) * (counts[i + j] - mu2)
        ac[j] = s / (var * m)

    return ac

"""
4. The binned ACF is still computed with 5 ms bins, but it is read out at the iSTTC lag points (10 ms spacing, up to 6 s).
Both estimators now see an identical lag vector.
If the recording can't support the full window, the neuron is dropped instead of shortening the window. 
The same applies if iSTTC returns fewer than 600 lags.
"""
def calculate_binned_acf_on_grid(spike_times, total_duration, target_lags_s,
                                 bin_size=ACF_BIN_S):
    """
    Binned ACF evaluated EXACTLY at `target_lags_s` (seconds), i.e. the lag
    grid that iSTTC used. Returns (lags_s, ac). Returns empty arrays if the
    recording is too short to support the full window (no silent shortening).
    """
    empty = (np.array([]), np.array([]))
    target = np.asarray(target_lags_s, dtype=float)
    if len(spike_times) == 0 or len(target) == 0:
        return empty

    idx = np.rint(target / bin_size).astype(int)
    if not np.allclose(idx * bin_size, target, atol=1e-9):
        raise ValueError("iSTTC lag grid is not a multiple of the ACF bin size.")

    nlags = int(idx.max()) + 1
    n_bins = int(total_duration / bin_size)
    if n_bins < nlags + 5:
        return empty

    counts, _ = np.histogram(spike_times, bins=n_bins, range=(0, n_bins * bin_size))
    ac = _binned_acf_nb(counts.astype(np.float64), nlags)
    return target, ac[idx]


# =============================================================================
# PART 2: PER-NEURON DUAL ESTIMATION
# =============================================================================

_UNIT_CHECK_DONE = False

"""1. this part is for your first feedback: - It looks like the isttc pipeline (at least your version on zenodo) outputs in milliseconds, but then you multiply by 1000.0 in your notebook anyway. 
It would be more clear to call the output of isttc tau_effective_ms so units are explicit 
iSTTC lags are converted to seconds immediately.
Both estimators go through the same _fit_to_tau_ms, so the * 1000 happens in one place with a known unit.
The detected unit is printed once ([units] ...)
"""
def _lags_to_seconds(lags):
    """isttc_pipeline may return ms; force seconds. Expected max ~ 6 s (6000 ms)."""
    global _UNIT_CHECK_DONE
    lags = np.asarray(lags, dtype=float)
    if len(lags) and lags.max() > 100:
        lags = lags / 1000.0
        if not _UNIT_CHECK_DONE:
            print("  [units] calculate_isttc returned lags in ms -> converted to s")
    elif not _UNIT_CHECK_DONE:
        print("  [units] calculate_isttc returned lags in s")
    _UNIT_CHECK_DONE = True
    return lags


def _fit_to_tau_ms(lags_s, vals):
    """Fit on a lag vector in SECONDS; return (tau_effective_ms, r2) or (nan, nan)."""
    fit = fit_multi_exponential(lags_s, vals)
    r2 = fit.get("r_squared", np.nan)
    tau_s = fit.get("tau_effective", np.nan)
    if r2 is None or tau_s is None or np.isnan(r2) or np.isnan(tau_s):
        return np.nan, np.nan
    if r2 < MIN_R2_FIT:
        return np.nan, np.nan
    return float(tau_s) * 1000.0, float(r2)


def process_neuron_dual(spike_times, total_duration):
    """Both estimators, same spike train, same lag grid, same fit routine."""
    out = {"tau_isttc_ms": np.nan, "tau_binned_ms": np.nan,
           "r2_isttc": np.nan, "r2_binned": np.nan,
           "n_spikes": len(spike_times)}

    if len(spike_times) < MIN_SPIKES:
        return out

    # --- iSTTC ---
    lags_i, vals_i = calculate_isttc(spike_times, total_duration,
                                     delta_t=ISTTC_DELTA_T_S, nlags=N_LAGS,
                                     lag_shift=LAG_STEP_S)
    lags_i = _lags_to_seconds(lags_i)
    vals_i = np.asarray(vals_i, dtype=float)
    if len(lags_i) < N_LAGS:      # full window required
        return out

    out["tau_isttc_ms"], out["r2_isttc"] = _fit_to_tau_ms(lags_i, vals_i)

    # --- binned ACF, on the identical lag grid ---
    lags_b, vals_b = calculate_binned_acf_on_grid(spike_times, total_duration, lags_i)
    if len(lags_b) == N_LAGS and not np.isnan(vals_b).any():
        out["tau_binned_ms"], out["r2_binned"] = _fit_to_tau_ms(lags_b, vals_b)

    return out


def get_beryl_regions_for_clusters(clusters, channels):
    from iblatlas.atlas import BrainRegions

    br = BrainRegions()
    chan_ids = np.asarray(channels["brainLocationIds_ccf_2017"])
    clu_channels = np.asarray(clusters["channels"])
    beryl_acronyms_per_channel = br.id2acronym(chan_ids, mapping="Beryl")
    return beryl_acronyms_per_channel[clu_channels]

"""
3. The label distribution and the number of good neurons are printed per probe.
"""
def _get_qc_labels(clusters, n_clusters):
    """IBL QC labels. No silent fallback: raise if they cannot be found."""
    labels = None
    if "label" in clusters:
        labels = np.asarray(clusters["label"], dtype=float)
    elif "metrics" in clusters and "label" in getattr(clusters["metrics"], "columns", []):
        labels = np.asarray(clusters["metrics"]["label"], dtype=float)
    if labels is None:
        raise RuntimeError("No QC 'label' found in clusters (nor in clusters['metrics']). "
                           "Refusing to continue without QC.")
    if len(labels) != n_clusters:
        raise RuntimeError(f"QC label length {len(labels)} != n_clusters {n_clusters}")
    return labels


def run_dual_estimation(one, pids, verbose=True, n_jobs=-1, apply_ibl_qc=True,
                        registry_csv=REGISTRY_CSV, csv_path=CSV_PATH):
    """
    pids: list of (eid, pid, probe_name).
    Returns a DataFrame: session_id, probe, pid, cluster_id, region, epoch_source,
    tau_isttc_ms, tau_binned_ms, r2_isttc, r2_binned, n_spikes, (+ subject).
    """
    all_rows = []

    for eid, pid, probe_name in pids:
        if verbose:
            print(f"\n{'=' * 65}")
            print(f"  Dual estimator | Session: {eid} | Probe: {probe_name}")
            print(f"{'=' * 65}")

        # ---- 1. Epoch window ----
        t_start, t_end, epoch_source = get_spontaneous_window(one, eid, verbose=verbose)
        if t_start is None:
            print(f"  SKIP: no spontaneous/passive epoch for {eid}.")
            continue  
        """2. this part is new and your review was: I’m concerned about get_spontaneous_window; 
        it actually seems to try to load task-related activity, and only the passive period if that fails 
        epoch_source is also saved as a column for every neuron.
        Windows shorter than 12 s (twice the 6 s fit window) are skipped.
        """
        if REQUIRE_PASSIVE and "passive" not in str(epoch_source).lower():
            print(f"SKIP: epoch source is '{epoch_source}', not passive ({eid}).")
            continue
        recording_duration = t_end - t_start
        if recording_duration < N_LAGS * LAG_STEP_S * 2:
            print(f"  SKIP: window {recording_duration:.1f} s too short for the 6 s fit window.")
            continue

        # ---- 2. Load ----
        try:
            spikes = one.load_object(eid, 'spikes', collection=f'alf/{probe_name}/pykilosort')
            clusters = one.load_object(eid, 'clusters', collection=f'alf/{probe_name}/pykilosort')
            channels = one.load_object(eid, 'channels', collection=f'alf/{probe_name}/pykilosort')
        except Exception as e:
            print(f"  SKIP: could not load spikes/clusters/channels: {e}")
            continue

        # ---- 3. Regions ----
        try:
            cluster_regions = get_beryl_regions_for_clusters(clusters, channels)
        except Exception as e:
            print(f"  SKIP: could not assign regions: {e}")
            continue

        # ---- 4. QC (strict, no silent failure) ----
        n_clusters = len(clusters['channels'])
        if apply_ibl_qc:
            qc_labels = _get_qc_labels(clusters, n_clusters)
            vals, cnts = np.unique(qc_labels, return_counts=True)
            if verbose:
                print(f"  QC label distribution: {dict(zip(vals.round(3), cnts))}")
            good = qc_labels == IBL_GOOD_LABEL
            if verbose:
                print(f"  Good (label == {IBL_GOOD_LABEL}): {good.sum()} / {n_clusters}")
        else:
            good = np.ones(n_clusters, dtype=bool)

        # ---- 5. Spikes in window ----
        spike_times_all = np.asarray(spikes['times'])
        spike_clusters_all = np.asarray(spikes['clusters'])
        sp_mask = (spike_times_all >= t_start) & (spike_times_all <= t_end)
        spont_times = spike_times_all[sp_mask] - t_start
        spont_clusters = spike_clusters_all[sp_mask]

        cluster_ids = np.arange(n_clusters)
        spike_dict = {cid: spont_times[spont_clusters == cid] for cid in cluster_ids}

        # ---- 6. Both estimators per good cluster ----
        def _one_cluster(cid):
            if not good[cid]:
                return None
            st = spike_dict.get(cid)
            if st is None or len(st) < MIN_SPIKES:
                return None
            res = process_neuron_dual(st, recording_duration)
            if np.isnan(res["tau_isttc_ms"]) and np.isnan(res["tau_binned_ms"]):
                return None
            res.update({"session_id": eid, "probe": probe_name, "pid": pid,
                        "cluster_id": int(cid), "region": cluster_regions[cid],
                        "epoch_source": str(epoch_source)})
            return res

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_cluster)(cid) for cid in tqdm(cluster_ids, desc="  Neurons", ncols=80)
        )
        results = [r for r in results if r is not None]
        if verbose:
            print(f"  Neurons with >=1 valid estimate: {len(results)}")
        all_rows.extend(results)

    if not all_rows:
        print("No usable neurons collected across any session.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # ---- 7. Subject identity ----
    if os.path.exists(registry_csv):
        reg = pd.read_csv(registry_csv)
        if "eid" in reg.columns and "session_id" not in reg.columns:
            reg = reg.rename(columns={"eid": "session_id"})
        reg["subject"] = reg["subject"].astype(str).str.replace('KS_091', 'KS091', regex=False).str.strip()
        reg_keep = reg[["pid", "session_id", "subject"]].drop_duplicates()
        df = df.merge(reg_keep, on=["pid", "session_id"], how="left")
        n_missing = df["subject"].isna().sum()
        if n_missing:
            print(f"  WARNING: {n_missing} neurons have no subject in {registry_csv}")
    else:
        print(f"  WARNING: {registry_csv} not found -> no 'subject' column.")

    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} neurons to {csv_path}")
    return df


# =============================================================================
# PART 3: COMMON NEURON SET (built ONCE, used by all 8 cells)
# =============================================================================

TAU_COLS = {"binned_ACF": "tau_binned_ms", "iSTTC": "tau_isttc_ms"}

"""
6. The common set is built once, with region counts computed on that set.
All 8 cells use the same d, and this is enforced:
"""

def build_common_set(df, region_col="region", subject_col="subject",
                     session_col="session_id", pid_col="pid",
                     min_neurons_per_region=MIN_NEURONS_PER_REGION):
    needed = list(TAU_COLS.values()) + [region_col, subject_col, session_col, pid_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = df.dropna(subset=needed).copy()
    d = d[(d["tau_binned_ms"] > 0) & (d["tau_isttc_ms"] > 0)]

    counts = d[region_col].value_counts()
    keep_regions = counts[counts >= min_neurons_per_region].index
    d = d[d[region_col].isin(keep_regions)].copy()

    for c in [subject_col, session_col, pid_col, region_col]:
        d[c] = d[c].astype(str).astype("category")

    print("\n" + "=" * 70)
    print("COMMON NEURON SET (identical for all 8 cells)")
    print("=" * 70)
    print(f"  input rows                     : {len(df)}")
    print(f"  with BOTH estimators valid     : {len(d)}")
    print(f"  regions (>= {min_neurons_per_region} neurons)     : {d[region_col].nunique()}")
    print(f"  subjects / sessions / probes   : "
          f"{d[subject_col].nunique()} / {d[session_col].nunique()} / {d[pid_col].nunique()}")
    if len(d) < 50 or d[region_col].nunique() < 2:
        raise RuntimeError("Common set too small for the analysis.")
    return d


# =============================================================================
# PART 4: THE TWO MODELS
# =============================================================================

def _transform(d, tau_col, scale):
    if scale == "log":
        return np.log10(d[tau_col].to_numpy(dtype=float))
    if scale == "raw":
        return d[tau_col].to_numpy(dtype=float)
    raise ValueError(scale)


def ordinary_r2(d, tau_col, scale, region_col="region"):
    """Region-only dummy regression R^2 on the common set."""
    from sklearn.linear_model import LinearRegression

    y = _transform(d, tau_col, scale)
    X = pd.get_dummies(d[region_col], drop_first=True).to_numpy(dtype=float)
    y_hat = LinearRegression().fit(X, y).predict(X)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _find_sd_var(posterior, factor):
    """Bambi names group-level SDs '1|<factor>_sigma'."""
    exact = f"1|{factor}_sigma"
    if exact in posterior.data_vars:
        return exact
    cands = [v for v in posterior.data_vars
             if factor in v and v.lower().endswith("sigma")]
    if len(cands) != 1:
        raise KeyError(f"Cannot identify SD variable for '{factor}': {cands}")
    return cands[0]


def hierarchical_variance_share(d, tau_col, scale, region_col="region",
                                subject_col="subject", session_col="session_id",
                                pid_col="pid", draws=1000, tune=1000, chains=4):
    """
    Variance share per component, computed PER POSTERIOR DRAW
    (var_k / sum_j var_j for each draw), then averaged over draws.
    """
    import bambi as bmb

    data = d[[region_col, subject_col, session_col, pid_col]].copy()
    data["y"] = _transform(d, tau_col, scale)

    formula = (f"y ~ 1 + (1|{region_col}) + (1|{subject_col})"
               f" + (1|{session_col}) + (1|{pid_col})")
    model = bmb.Model(formula, data=data, family="gaussian")
    idata = model.fit(draws=draws, tune=tune, chains=chains, cores=chains,
                      target_accept=0.9, random_seed=42)
    post = idata.posterior

    factors = [region_col, subject_col, session_col, pid_col]
    var = {}
    for f in factors:
        var[f] = post[_find_sd_var(post, f)].values.reshape(-1) ** 2
    var["residual"] = post["sigma"].values.reshape(-1) ** 2


"""
7. The proportion is computed per posterior draw first, then averaged over draws, as Jason suggested. Previously it was a ratio of means.
SD variables are looked up by their Bambi names (1|region_sigma etc.) instead of the "sigma" in name search.
The old region_key = [k for k in hshare if region_col in k] could match the n_regions key, so it was removed.
A 94% interval for the region share is now reported too.
"""
    names = list(var)
    mat = np.stack([var[k] for k in names])            # (components, draws)
    share_draws = mat / mat.sum(axis=0, keepdims=True)  # per-draw proportions

    out = {k: float(share_draws[i].mean()) for i, k in enumerate(names)}
    lo, hi = np.percentile(share_draws[0], [3, 97])     # region interval
    out["region_ci_lo"], out["region_ci_hi"] = float(lo), float(hi)
    out["region_share"] = out[region_col]
    return out


# =============================================================================
# PART 5: 2 x 2 x 2 TABLE (estimator x model x scale)
# =============================================================================

def build_8cell_table(d, region_col="region", subject_col="subject",
                      session_col="session_id", pid_col="pid", out_csv="eight_cell_table.csv"):
    rows = []
    n_ref = len(d)
    k_ref = d[region_col].nunique()

    """
The log_transform=True/False flag is replaced by scale="raw"/"log".
The result is 2 x 2 x 2 = 8 cells, saved to eight_cell_table.csv.
build_scale_crossed_diagnostic is removed because it is now redundant.
    """
    for estimator, tau_col in TAU_COLS.items():
        for scale in ["raw", "log"]:
            # --- ordinary R^2 ---
            r2 = ordinary_r2(d, tau_col, scale, region_col)
            rows.append(dict(estimator=estimator, model="ordinary_R2", scale=scale,
                             value=r2, ci_lo=np.nan, ci_hi=np.nan,
                             n_neurons=len(d), n_regions=d[region_col].nunique()))

            # --- hierarchical variance share ---
            print(f"\n>>> hierarchical fit: {estimator} / {scale}")
            h = hierarchical_variance_share(d, tau_col, scale, region_col,
                                            subject_col, session_col, pid_col)
            rows.append(dict(estimator=estimator, model="hierarchical_share", scale=scale,
                             value=h["region_share"], ci_lo=h["region_ci_lo"],
                             ci_hi=h["region_ci_hi"],
                             n_neurons=len(d), n_regions=d[region_col].nunique()))

    res = pd.DataFrame(rows)

    # same-neuron guarantee
    assert (res["n_neurons"] == n_ref).all() and (res["n_regions"] == k_ref).all(), \
        "Neuron/region sets differ across cells!"

    res.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print(f"REGION EFFECT, estimator x model x scale   (n = {n_ref} neurons, {k_ref} regions, all cells)")
    print("=" * 78)
    print(f"{'':32s} {'binned ACF':>20s} {'iSTTC':>20s}")
    for model in ["ordinary_R2", "hierarchical_share"]:
        for scale in ["raw", "log"]:
            vals = []
            for est in TAU_COLS:
                v = res[(res.estimator == est) & (res.model == model) & (res.scale == scale)]["value"].iloc[0]
                vals.append(v * 100)
            print(f"{model + ' (' + scale + ')':32s} {vals[0]:>19.1f}% {vals[1]:>19.1f}%")
    print("=" * 78)
    print("Reading guide:")
    print("  - columns (fixed row)      -> ESTIMATOR effect")
    print("  - ordinary vs hierarchical -> MODEL effect (fixed estimator + scale)")
    print("  - raw vs log               -> SCALE effect (fixed estimator + model)")
    print(f"Saved {out_csv}")
    return res


# =============================================================================
# ENTRY POINT
# =============================================================================

PIDS = [
    ('fa704052-147e-46f6-b190-a65b837e605e', '00a824c0-e060-495f-9ebc-79c82fef4c67', 'probe00'),
    ('1b9e349e-93f2-41cc-a4b5-b212d7ddc8df', '3c283107-7012-48fc-a6c2-ed096b23974f', 'probe01'),  # SCm
    ('03cf52f6-fba6-4743-a42e-dd1ac3072343', 'a3d13b05-bf4d-427a-a2d5-2fe050d603ec', 'probe00'),  # SUB
    ('8c552ddc-813e-4035-81cc-3971b57efe65', '10ef1dcd-093c-4839-8f38-90a25edefb49', 'probe01'),  # PAG
    ('1a507308-c63a-4e02-8f32-3239a07dc578', '7be00744-2e27-4062-8f56-3969e24e9990', 'probe00'),  # VPM
    ('e5fae088-ed96-4d9b-82f9-dfd13c259d52', '071f02e7-752a-4094-af79-8dd764e9d85d', 'probe00'),  # PO
    ('1928bf72-2002-46a6-8930-728420402e01', '68f06c5f-8566-4a4f-a4b1-ab8398724913', 'probe01'),  # MDRN
    ('dc21e80d-97d7-44ca-a729-a8e3f9b14305', '176b4fe3-f570-4d9f-9e25-a5d218f75c8b', 'probe00'),  # PAR
]


def main():
    if "--reuse" in sys.argv and os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"Reusing {CSV_PATH} ({len(df)} neurons)")
    else:
        from one.api import ONE
        one = ONE(base_url='https://openalyx.internationalbrainlab.org',
                  username='intbrainlab', password='international', silent=True)
        df = run_dual_estimation(one, PIDS)

    if len(df) == 0:
        return

    d = build_common_set(df)
    build_8cell_table(d)


if __name__ == "__main__":
    main()