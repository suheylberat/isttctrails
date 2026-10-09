"""
isttc_pipeline.py
------------------
iSTTC intrinsic timescale analysis pipeline.
Adapted from Google Colab notebook for local execution.

Requirements:
    pip install ONE-api pandas numpy scipy matplotlib seaborn tqdm numba joblib

Usage:
    python isttc_pipeline.py

    Or import and call directly:
        from isttc_pipeline import run_sessions, analyze_session
        from one.api import ONE
        one = ONE(base_url='https://openalyx.internationalbrainlab.org',
                  username='intbrainlab', password='international', silent=True)
        results = run_sessions(one, pids=[('eid1', 'pid1', 'probe00'), ...])
"""

# -- Imports ------------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS  —  edit these to match your local setup
# =============================================================================
OUTPUT_CSV    = 'good_isttc.csv'
OUTPUT_DIR    = 'isttc_summary_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# CORE iSTTC FUNCTIONS
# =============================================================================

@njit(cache=True)
def _tiling_proportion_nb(spike_times, start, end, delta_t, total_duration, shift):
    """T_A with slice bounds and shift - no array allocation."""
    n = end - start
    if n == 0:
        return 0.0
    cum_max_end = spike_times[start] - shift + delta_t
    if cum_max_end > total_duration:
        cum_max_end = total_duration
    s0 = spike_times[start] - shift - delta_t
    if s0 < 0.0:
        s0 = 0.0
    total = cum_max_end - s0
    for i in range(start + 1, end):
        s = spike_times[i] - shift - delta_t
        if s < 0.0:
            s = 0.0
        e = spike_times[i] - shift + delta_t
        if e > total_duration:
            e = total_duration
        overlap = cum_max_end - s
        if overlap < 0.0:
            overlap = 0.0
        total += (e - s) - overlap
        if e > cum_max_end:
            cum_max_end = e
    if total < 0.0:
        total = 0.0
    if total > total_duration:
        total = total_duration
    return total / total_duration


@njit(cache=True)
def _spikes_in_tiling_nb(spike_times_test, t_start, t_end, shift_test,
                          spike_times_ref,  r_start, r_end,  shift_ref,
                          delta_t):
    """P_A|B: fraction of (test-shift_test) spikes within delta_t of (ref-shift_ref) spikes."""
    n_test = t_end - t_start
    if n_test == 0 or (r_end - r_start) == 0:
        return 0.0
    count = 0
    for i in range(t_start, t_end):
        val = spike_times_test[i] - shift_test
        lo  = val - delta_t + shift_ref
        hi  = val + delta_t + shift_ref
        left = r_start
        right = r_end
        while left < right:
            mid = (left + right) // 2
            if spike_times_ref[mid] < lo:
                left = mid + 1
            else:
                right = mid
        l2 = r_start
        r2 = r_end
        while l2 < r2:
            mid = (l2 + r2) // 2
            if spike_times_ref[mid] <= hi:
                l2 = mid + 1
            else:
                r2 = mid
        if l2 > left:
            count += 1
    return count / n_test


@njit(cache=True)
def _isttc_nb(spike_times, total_duration, delta_t, nlags, lag_shift):
    """iSTTC lag loop - fully compiled, zero array allocation inside loop."""
    isttc_values = np.full(nlags, np.nan)
    isttc_values[0] = 1.0
    n  = len(spike_times)
    si = 0
    ei = n
    for i in range(1, nlags):
        lag = i * lag_shift
        dur = total_duration - lag
        while si < n and spike_times[si] < lag:
            si += 1
        while ei > 0 and spike_times[ei - 1] > total_duration - lag:
            ei -= 1
        if (n - si) < 2 or ei < 2:
            continue
        TA = _tiling_proportion_nb(spike_times, si, n,  delta_t, dur, lag)
        TB = _tiling_proportion_nb(spike_times, 0,  ei, delta_t, dur, 0.0)
        PA = _spikes_in_tiling_nb(spike_times, si, n,  lag,
                                   spike_times, 0,  ei, 0.0, delta_t)
        PB = _spikes_in_tiling_nb(spike_times, 0,  ei, 0.0,
                                   spike_times, si, n,  lag, delta_t)
        denom1 = 1.0 - PA * TB
        denom2 = 1.0 - PB * TA
        if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
            continue
        isttc_values[i] = 0.5 * ((PA - TB) / denom1 + (PB - TA) / denom2)
    return isttc_values


def calculate_isttc(spike_times, total_duration,
                    delta_t=0.005, nlags=600, lag_shift=0.01):
    """iSTTC autocorrelation function. (Pochinok et al. 2026, Sec. 7.4)"""
    if len(spike_times) == 0:
        return np.array([]), np.array([])
    spike_times = np.ascontiguousarray(np.sort(spike_times), dtype=np.float64)
    lags = np.arange(nlags) * lag_shift
    isttc_values = _isttc_nb(spike_times, float(total_duration),
                              float(delta_t), int(nlags), float(lag_shift))
    return lags, isttc_values


# Warmup JIT on import
_dummy = np.ascontiguousarray(np.sort(np.random.uniform(0, 10, 20)), dtype=np.float64)
_ = _isttc_nb(_dummy, 10.0, 0.005, 10, 0.005)
print("✓ Numba JIT compiled")


# =============================================================================
# MULTI-TIMESCALE FIT (Shi et al. 2025)
# =============================================================================

def _multi_exp(t, *params):
    """M-component exponential: f(t) = Σ ci·exp(-t/τi)"""
    n = len(params) // 2
    result = np.zeros_like(t, dtype=float)
    for i in range(n):
        result += params[2*i] * np.exp(-t / params[2*i+1])
    return result


def fit_multi_exponential(lags, isttc_values, start_lag_idx=1,
                           max_components=4, min_component_fraction=0.01):
    """
    1 to 4-component exponential fit + BIC model selection.
    Returns: tau_effective (Shi et al. Eq. 3), individual tau/c, BIC, R², CI
    """
    valid = ~np.isnan(isttc_values)
    lags_fit = lags[valid][start_lag_idx:]
    vals_fit = isttc_values[valid][start_lag_idx:]

    empty = {
        'n_timescales': np.nan,
        'tau_1': np.nan, 'tau_2': np.nan, 'tau_3': np.nan, 'tau_4': np.nan,
        'c_1':   np.nan, 'c_2':   np.nan, 'c_3':   np.nan, 'c_4':   np.nan,
        'tau_effective': np.nan, 'r_squared': np.nan, 'bic_values': {},
        'fit_values': np.full_like(lags, np.nan),
        'ci_lower_ms': np.nan, 'ci_upper_ms': np.nan, 'ci_excludes_zero': False,
        'tau_at_bound': False,
    }

    if len(lags_fit) < 4:
        return empty

    n_data = len(lags_fit)
    best_bic, best_result, bic_values = np.inf, None, {}

    tau_lower = 0.005
    tau_upper = max(float(lags_fit[-1]), 0.5)

    for M in range(1, max_components + 1):
        n_params = 2 * M
        if n_data <= n_params + 1:
            break

        amp_init = max(np.max(np.abs(vals_fit)) / M, 1e-6)
        tau_init = np.geomspace(tau_lower, tau_upper, M)
        p0 = sum([[amp_init, tau_init[j]] for j in range(M)], [])
        lb = sum([[0.0, tau_lower] for _ in range(M)], [])
        ub = sum([[np.inf, tau_upper] for _ in range(M)], [])

        try:
            popt, pcov = curve_fit(_multi_exp, lags_fit, vals_fit,
                                   p0=p0, bounds=(lb, ub),
                                   maxfev=20000, method='trf')
        except Exception:
            continue

        c_vals, tau_vals = popt[0::2], popt[1::2]
        c_sum = np.sum(c_vals)
        if c_sum <= 0 or np.any(c_vals / c_sum < min_component_fraction):
            continue
        if np.any(c_vals > 10.0):
            continue

        fitted = _multi_exp(lags_fit, *popt)
        ss_res = np.sum((vals_fit - fitted) ** 2)
        ss_tot = np.sum((vals_fit - np.mean(vals_fit)) ** 2)
        r2  = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        bic = n_data * np.log(ss_res / n_data + 1e-30) + n_params * np.log(n_data)
        bic_values[M] = bic

        if bic < best_bic:
            best_bic = bic
            tau_eff = float(np.sum(c_vals * tau_vals) / c_sum)

            try:
                grad = sum([[(tau_vals[j] - tau_eff) / c_sum,
                              c_vals[j] / c_sum] for j in range(M)], [])
                grad = np.array(grad)
                std_tau_eff = np.sqrt(max(grad @ pcov @ grad, 0.0))
                t_crit = stats.t.ppf(0.975, max(n_data - n_params, 1))
                ci_lo  = (tau_eff - t_crit * std_tau_eff) * 1000
                ci_hi  = (tau_eff + t_crit * std_tau_eff) * 1000
                ci_ex0 = bool(ci_lo > 0)
            except Exception:
                ci_lo, ci_hi, ci_ex0 = np.nan, np.nan, bool(tau_eff > 0)

            sort_idx   = np.argsort(tau_vals)
            tau_sorted = tau_vals[sort_idx] * 1000
            c_sorted   = c_vals[sort_idx]
            tau_at_bound = bool(np.any(tau_vals >= tau_upper * 0.99))

            best_result = {
                'n_timescales': M,
                'tau_1': float(tau_sorted[0]) if M >= 1 else np.nan,
                'tau_2': float(tau_sorted[1]) if M >= 2 else np.nan,
                'tau_3': float(tau_sorted[2]) if M >= 3 else np.nan,
                'tau_4': float(tau_sorted[3]) if M >= 4 else np.nan,
                'c_1':   float(c_sorted[0])   if M >= 1 else np.nan,
                'c_2':   float(c_sorted[1])   if M >= 2 else np.nan,
                'c_3':   float(c_sorted[2])   if M >= 3 else np.nan,
                'c_4':   float(c_sorted[3])   if M >= 4 else np.nan,
                'tau_effective': tau_eff * 1000,
                'r_squared': float(r2),
                'bic_values': bic_values,
                'fit_values': _multi_exp(lags, *popt),
                'ci_lower_ms': float(ci_lo),
                'ci_upper_ms': float(ci_hi),
                'ci_excludes_zero': ci_ex0,
                'tau_at_bound': tau_at_bound,
            }

    if best_result is None:
        return empty

    best_result['bic_values'] = bic_values
    return best_result


# =============================================================================
# AUXILIARY METRICS
# =============================================================================

def compute_local_variance(spike_times):
    """Local Variance (Lv) - Shinomoto et al. 2009."""
    if len(spike_times) < 3:
        return np.nan
    isi = np.diff(np.sort(spike_times))
    if len(isi) < 2:
        return np.nan
    return float(3.0 * np.mean(
        ((isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:] + 1e-30)) ** 2
    ))


# =============================================================================
# QUALITY CRITERIA
# =============================================================================

MIN_SPIKE_COUNT = 100

def check_quality_criteria(lags, isttc_values, fit_results, spike_count=None):
    q = {
        'acf_decline_50_200ms':  False,
        'ci_excludes_zero':      bool(fit_results.get('ci_excludes_zero', False)),
        'r_squared_above_0p5':   False,
        'spike_count_above_min': False,
        'tau_at_bound':          bool(fit_results.get('tau_at_bound', False)),
        'passed_quality_filter': False,
    }

    mask = (lags >= 0.05) & (lags <= 0.20)
    valid_vals = isttc_values[mask][~np.isnan(isttc_values[mask])]
    if len(valid_vals) >= 2:
        q['acf_decline_50_200ms'] = bool(valid_vals[0] > valid_vals[-1])

    r2 = fit_results.get('r_squared', np.nan)
    if not np.isnan(r2):
        q['r_squared_above_0p5'] = bool(r2 >= 0.5)

    if spike_count is not None:
        q['spike_count_above_min'] = bool(spike_count >= MIN_SPIKE_COUNT)

    q['passed_quality_filter'] = (
        q['acf_decline_50_200ms']  and
        q['ci_excludes_zero']      and
        q['r_squared_above_0p5']   and
        q['spike_count_above_min'] and
        not q['tau_at_bound']
    )
    return q


# =============================================================================
# SINGLE NEURON ANALYSIS
# =============================================================================

def analyze_single_neuron(spike_times, recording_duration,
                           neuron_id='Neuron',
                           delta_t=0.005, nlags=600, lag_shift=0.01,
                           max_components=4):
    spike_times  = np.sort(np.asarray(spike_times, dtype=float))
    firing_rate  = len(spike_times) / recording_duration if recording_duration > 0 else np.nan
    spike_count  = len(spike_times)
    lv           = compute_local_variance(spike_times)

    lags, isttc_values = calculate_isttc(
        spike_times, recording_duration,
        delta_t=delta_t, nlags=nlags, lag_shift=lag_shift
    )
    fit     = fit_multi_exponential(lags, isttc_values, max_components=max_components)
    quality = check_quality_criteria(lags, isttc_values, fit, spike_count=spike_count)

    return {
        'neuron_id':           neuron_id,
        'n_spikes':            spike_count,
        'firing_rate_hz':      firing_rate,
        'local_variance_lv':   lv,
        'n_timescales':        fit['n_timescales'],
        'tau_1_ms':            fit['tau_1'],
        'tau_2_ms':            fit['tau_2'],
        'tau_3_ms':            fit['tau_3'],
        'tau_4_ms':            fit['tau_4'],
        'c_1': fit['c_1'], 'c_2': fit['c_2'],
        'c_3': fit['c_3'], 'c_4': fit['c_4'],
        'tau_effective_ms':    fit['tau_effective'],
        'r_squared':           fit['r_squared'],
        'ci_lower_ms':         fit['ci_lower_ms'],
        'ci_upper_ms':         fit['ci_upper_ms'],
        'bic_1': fit['bic_values'].get(1, np.nan),
        'bic_2': fit['bic_values'].get(2, np.nan),
        'bic_3': fit['bic_values'].get(3, np.nan),
        'bic_4': fit['bic_values'].get(4, np.nan),
        **quality,
        '_lags':         lags,
        '_isttc_values': isttc_values,
        '_fit_values':   fit['fit_values'],
    }


# =============================================================================
# SESSION HELPERS
# =============================================================================

def get_spontaneous_window(one, eid, verbose=True):
    """Find the passive/spontaneous epoch for a session."""
    try:
        trials = one.load_object(eid, 'trials', collection='alf')
        t_start = float(trials['intervals'][0, 0])
        t_end   = float(trials['intervals'][-1, 1])
        if verbose:
            print(f"  Epoch: trials [{t_start:.1f}s, {t_end:.1f}s]")
        return t_start, t_end, 'trials'
    except Exception:
        pass
    try:
        passive = one.load_object(eid, 'passivePeriods', collection='alf')
        t_start = float(passive['spontaneousActivity'][0])
        t_end   = float(passive['spontaneousActivity'][1])
        if verbose:
            print(f"  Epoch: spontaneousActivity [{t_start:.1f}s, {t_end:.1f}s]")
        return t_start, t_end, 'spontaneousActivity'
    except Exception:
        pass
    if verbose:
        print("  No passive epoch found.")
    return None, None, None


def _process_cluster(cluster_id, spike_dict, recording_duration,
                     qc_labels, apply_ibl_qc, min_spikes,
                     eid, probe, epoch_source, t_start, t_end):
    """Worker function for parallel cluster processing."""
    if apply_ibl_qc and qc_labels is not None:
        try:
            if qc_labels[cluster_id] < 2:
                return None
        except Exception:
            pass

    spike_times = spike_dict.get(cluster_id)
    if spike_times is None or len(spike_times) < min_spikes:
        return None

    result = analyze_single_neuron(
        spike_times,
        recording_duration=recording_duration,
        neuron_id=f'Cluster_{cluster_id}',
    )

    return {
        'session_id':   eid,
        'probe':        probe,
        'cluster_id':   int(cluster_id),
        'epoch_source': epoch_source,
        't_start_s':    t_start,
        't_end_s':      t_end,
        'brain_region': np.nan,
        'beryl_region': np.nan,
        'ap_um':        np.nan,
        'dv_um':        np.nan,
        'ml_um':        np.nan,
        'n_spikes_spontaneous': result['n_spikes'],
        'firing_rate_hz':       result['firing_rate_hz'],
        'local_variance_lv':    result['local_variance_lv'],
        'n_timescales':         result['n_timescales'],
        'tau_1_ms':             result['tau_1_ms'],
        'tau_2_ms':             result['tau_2_ms'],
        'tau_3_ms':             result['tau_3_ms'],
        'tau_4_ms':             result['tau_4_ms'],
        'c_1': result['c_1'], 'c_2': result['c_2'],
        'c_3': result['c_3'], 'c_4': result['c_4'],
        'tau_effective_ms':     result['tau_effective_ms'],
        'r_squared':            result['r_squared'],
        'ci_lower_ms':          result['ci_lower_ms'],
        'ci_upper_ms':          result['ci_upper_ms'],
        'bic_1': result['bic_1'], 'bic_2': result['bic_2'],
        'bic_3': result['bic_3'], 'bic_4': result['bic_4'],
        'acf_decline_50_200ms':  result['acf_decline_50_200ms'],
        'ci_excludes_zero':      result['ci_excludes_zero'],
        'r_squared_above_0p5':   result['r_squared_above_0p5'],
        'spike_count_above_min': result['spike_count_above_min'],
        'tau_at_bound':          result['tau_at_bound'],
        'passed_quality_filter': result['passed_quality_filter'],
    }


# =============================================================================
# SESSION ANALYSIS
# =============================================================================

def analyze_session(eid, one, probe='probe00',
                    min_spikes=100,
                    apply_ibl_qc=True,
                    n_jobs=-1,
                    verbose=True,
                    preloaded=None):

    if verbose:
        print(f"\n{'='*65}")
        print(f"  iSTTC | Session: {eid} | Probe: {probe}")
        print(f"{'='*65}")

    if preloaded is not None:
        t_start      = preloaded['t_start']
        t_end        = preloaded['t_end']
        epoch_source = preloaded['epoch_source']
        spikes       = preloaded['spikes']
        clusters     = preloaded['clusters']
    else:
        t_start, t_end, epoch_source = get_spontaneous_window(one, eid, verbose=verbose)
        if t_start is None:
            print(f"  SKIP: No passive epoch found for {eid}.")
            return pd.DataFrame()
        try:
            spikes = one.load_object(eid, 'spikes',
                                     collection=f'alf/{probe}/pykilosort')
        except Exception as e:
            print(f"  Could not load spike data: {e}")
            return pd.DataFrame()
        try:
            clusters = one.load_object(eid, 'clusters',
                                       collection=f'alf/{probe}/pykilosort')
        except Exception:
            clusters = None

    recording_duration = t_end - t_start

    if verbose:
        print(f"  Epoch source      : {epoch_source}")
        print(f"  Duration          : {recording_duration:.1f}s ({recording_duration/60:.1f} min)")

    qc_labels = None
    if apply_ibl_qc and clusters is not None:
        try:
            qc_labels = np.array(clusters['label'])
        except Exception:
            pass

    spike_times_all    = np.array(spikes['times'])
    spike_clusters_all = np.array(spikes['clusters'])
    sp_mask            = (spike_times_all >= t_start) & (spike_times_all <= t_end)
    spont_times        = spike_times_all[sp_mask] - t_start
    spont_clusters     = spike_clusters_all[sp_mask]
    unique_clusters    = np.unique(spont_clusters)

    if verbose:
        print(f"  Spikes in SP window  : {sp_mask.sum():,}")
        print(f"  Clusters in SP window: {len(unique_clusters):,}")

    spike_dict = {cid: spont_times[spont_clusters == cid] for cid in unique_clusters}

    raw_results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_process_cluster)(
            cid, spike_dict,
            recording_duration, qc_labels, apply_ibl_qc, min_spikes,
            eid, probe, epoch_source, t_start, t_end,
        )
        for cid in tqdm(unique_clusters, desc='  Clusters', unit='cluster',
                        ncols=80, leave=True)
    )

    results_list = [r for r in raw_results if r is not None]
    df = pd.DataFrame(results_list)

    if verbose and len(df) > 0:
        n_pass  = df['passed_quality_filter'].sum()
        med_tau = df.loc[df['passed_quality_filter'], 'tau_effective_ms'].median()
        print(f"\n  Completed!")
        print(f"  Neurons analysed  : {len(df)}")
        print(f"  Passed QC         : {n_pass} ({100*n_pass/len(df):.1f}%)")
        print(f"  Median τ_eff      : {med_tau:.1f} ms")
        print(f"{'='*65}\n")

    return df


# =============================================================================
# BATCH RUNNER
# =============================================================================

def run_sessions(one, pids, output_csv=OUTPUT_CSV, verbose=True):
    """
    Run iSTTC analysis on a list of (eid, pid, probe_name) tuples.

    Parameters
    ----------
    one         : ONE instance
    pids        : list of (eid, pid, probe_name) tuples
    output_csv  : path to incrementally save QC-passing neurons
    verbose     : print progress

    Returns
    -------
    pd.DataFrame of QC-passing neurons
    """
    all_results = []
    skipped     = []

    for eid, pid, probe_name in pids:
        try:
            df = analyze_session(eid, one, probe_name, verbose=verbose)
            if len(df) > 0:
                df = df.copy()
                df['pid'] = pid
                all_results.append(df)

                qc_mask = (
                    df['passed_quality_filter']
                    .astype(str).str.strip().str.lower().eq('true')
                )
                good_session = df[qc_mask].copy()

                if len(good_session) > 0:
                    try:
                        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
                            existing = pd.read_csv(output_csv)
                            pd.concat([existing, good_session], ignore_index=True)\
                              .to_csv(output_csv, index=False)
                            print(f"  Saved {len(good_session)} neurons from {pid} "
                                  f"— running total: {len(existing) + len(good_session)}")
                        else:
                            good_session.to_csv(output_csv, index=False)
                            print(f"  Saved {len(good_session)} neurons from {pid} to new file.")
                    except Exception as save_err:
                        print(f"  SAVE ERROR — PID {pid}: {save_err}")
                else:
                    print(f"  PID {pid}: 0 neurons passed QC — nothing saved.")
            else:
                skipped.append(pid)

        except Exception as e:
            print(f"  ERROR — PID {pid} / session {eid}: {e}")
            skipped.append(pid)
            continue

    if skipped:
        print(f"\n  Skipped PIDs ({len(skipped)}): {skipped}")

    if not all_results:
        print("No results collected.")
        return pd.DataFrame()

    combined   = pd.concat(all_results, ignore_index=True)
    qc_mask    = (
        combined['passed_quality_filter']
        .astype(str).str.strip().str.lower().eq('true')
    )
    qc_good    = combined[qc_mask]
    n_total    = len(combined)
    n_qc       = len(qc_good)
    median_tau = qc_good['tau_effective_ms'].median() if n_qc > 0 else float('nan')

    print(f"\n{'='*65}")
    print("  BATCH COMPLETE")
    print(f"  Sessions processed : {len(all_results)} / {len(pids)}")
    print(f"  Total neurons      : {n_total}")
    print(f"  Passed QC          : {n_qc} ({100 * n_qc / n_total:.1f}%)")
    print(f"  Median τ_eff       : {median_tau:.1f} ms")
    print(f"  Output file        : {output_csv}")
    print(f"{'='*65}\n")

    return qc_good


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_neuron(result, title=None):
    lags_ms = result['_lags'] * 1000
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(lags_ms, result['_isttc_values'], 'o-', color='#2E86AB',
            markersize=3, linewidth=1.8, label='iSTTC', alpha=0.8)
    if not np.all(np.isnan(result['_fit_values'])):
        tau_eff = result['tau_effective_ms']
        n_tau   = result['n_timescales']
        r2      = result['r_squared']
        lbl = (f'{n_tau}-exp fit | τ_eff={tau_eff:.1f}ms | R²={r2:.3f}'
               if not np.isnan(tau_eff) else 'Fit')
        ax.plot(lags_ms, result['_fit_values'], 'r-', linewidth=2.2,
                label=lbl, alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.35)
    ax.axvspan(50, 200, alpha=0.07, color='orange', label='50-200ms QC')
    ax.set_xlabel('Lag (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('iSTTC', fontsize=12, fontweight='bold')
    ax.set_title(title or result['neuron_id'], fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    ax.axis('off')
    qc = {k: ('✓' if result.get(k) else '✗') for k in
          ['acf_decline_50_200ms', 'ci_excludes_zero',
           'r_squared_above_0p5', 'spike_count_above_min']}
    comp = ""
    for k in range(1, 5):
        tk = result.get(f'tau_{k}_ms', np.nan)
        ck = result.get(f'c_{k}', np.nan)
        if tk is not None and not np.isnan(float(tk)):
            comp += f"  τ{k}={tk:.1f}ms (c{k}={ck:.3f})\n"
    passed = '✓ PASSED' if result['passed_quality_filter'] else '✗ FAILED'
    text = (f"Neuron: {result['neuron_id']}\n"
            f"Spikes: {result['n_spikes']:,} | FR: {result['firing_rate_hz']:.2f}Hz | "
            f"Lv: {result['local_variance_lv']:.3f}\n\n"
            f"─── {result['n_timescales']} component(s) ───\n{comp}"
            f"τ_eff = {result['tau_effective_ms']:.2f} ms\n"
            f"CI: [{result['ci_lower_ms']:.1f}, {result['ci_upper_ms']:.1f}] ms\n"
            f"R² = {result['r_squared']:.4f}\n\n"
            f"{qc['acf_decline_50_200ms']} ACF decline (50-200ms)\n"
            f"{qc['ci_excludes_zero']} CI > 0\n"
            f"{qc['r_squared_above_0p5']} R² ≥ 0.5\n\n"
            f"{qc['spike_count_above_min']} Spike count\n"
            f"OVERALL: {passed}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#eaf4fb', alpha=0.5))
    plt.tight_layout()
    return fig


# =============================================================================
# ENTRY POINT  —  add your (eid, pid, probe_name) tuples to SESSIONS below
# =============================================================================

if __name__ == '__main__':
    from one.api import ONE

    # Add all session tuples here - (eid, pid, probe_name)
    SESSIONS = [
        # ('fa704052-147e-46f6-b190-a65b837e605e', '00a824c0-e060-495f-9ebc-79c82fef4c67', 'probe00'),
        # ('fa704052-147e-46f6-b190-a65b837e605e', '00a824c0-e060-495f-9ebc-79c82fef4c67', 'probe01'),
    ]

    if not SESSIONS:
        print("No sessions defined. Add (eid, pid, probe_name) tuples to SESSIONS.")
    else:
        one = ONE(
            base_url='https://openalyx.internationalbrainlab.org',
            username='intbrainlab',
            password='international',
            silent=True,
        )

        results = run_sessions(one, pids=SESSIONS, output_csv=OUTPUT_CSV)

        if len(results) > 0:
            print(f"\nDone. {len(results)} QC-passing neurons saved to {OUTPUT_CSV}")