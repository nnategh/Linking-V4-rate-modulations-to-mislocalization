"""Shared, leakage-safe PCA/GPR tuning helpers for cross-condition decoders."""

import os
import re
import copy

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.stats import rankdata, shapiro, ttest_rel, wilcoxon
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler


CONDITION_CONFIG = {
    "fixation": ("fixation", "fix", "SpikeProbe_fix", "Conditions_fix"),
    "peri_saccade": ("peri_saccade", "peri", "SpikeProbe_peri", "Conditions_peri"),
}


def parse_float_list(text):
    return [float(value) for value in text.split(",") if value.strip()]


def rank_available_probes_by_composite_error(
        trial_predictions_df, baseline_run, keys, available_probe_keys):
    """Rank available session/probe pairs without post-selection session filtering."""
    prediction_columns = keys + ["Regression_Pred_X", "Regression_Pred_Y"]
    baseline_trials = trial_predictions_df[
        trial_predictions_df["Run"] == baseline_run
    ][prediction_columns].copy()
    baseline_trials["Trial_Euclidean_Error"] = np.hypot(
        baseline_trials["Regression_Pred_X"] - baseline_trials["True_X"],
        baseline_trials["Regression_Pred_Y"] - baseline_trials["True_Y"],
    )
    probe_errors = baseline_trials.groupby(keys, as_index=False).agg(
        Fixation_Probe_Avg_Euclidean_Error=("Trial_Euclidean_Error", "mean"),
        Fixation_Probe_Num_Trials=("Trial_Euclidean_Error", "size"),
    )
    available_keys = available_probe_keys[keys].drop_duplicates()
    probe_errors = probe_errors.merge(
        available_keys, on=keys, how="inner", validate="one_to_one"
    )
    probe_errors = probe_errors[
        np.isfinite(probe_errors["Fixation_Probe_Avg_Euclidean_Error"])
    ].copy()
    if probe_errors.empty:
        raise ValueError("No finite fixation probe errors are available for composite ranking.")

    session_keys = [column for column in keys if column not in ("True_X", "True_Y")]
    probe_errors["Probe_Position_Percentile"] = probe_errors.groupby(
        ["True_X", "True_Y"]
    )["Fixation_Probe_Avg_Euclidean_Error"].rank(method="average", pct=True)

    other_probe_medians = pd.Series(index=probe_errors.index, dtype=float)
    for _, session_rows in probe_errors.groupby(session_keys, sort=False):
        percentiles = session_rows["Probe_Position_Percentile"]
        for row_index in session_rows.index:
            other_probe_medians.loc[row_index] = percentiles.drop(row_index).median()
    probe_errors["Other_Probe_Median_Percentile"] = other_probe_medians.fillna(1.0)
    probe_errors["Composite_Probe_Score"] = 0.5 * (
        probe_errors["Probe_Position_Percentile"]
        + probe_errors["Other_Probe_Median_Percentile"]
    )
    tie_break_columns = [
        "Composite_Probe_Score",
        "Probe_Position_Percentile",
        "Fixation_Probe_Avg_Euclidean_Error",
    ] + session_keys + ["True_X", "True_Y"]
    probe_errors = probe_errors.sort_values(tie_break_columns, kind="stable").reset_index(drop=True)
    probe_errors["Composite_Probe_Rank"] = np.arange(1, len(probe_errors) + 1)
    return probe_errors


def rank_available_probes_by_raw_error(
        trial_predictions_df, baseline_run, keys, available_probe_keys):
    """Rank available session/probe pairs by unadjusted mean Euclidean error."""
    prediction_columns = keys + ["Regression_Pred_X", "Regression_Pred_Y"]
    baseline_trials = trial_predictions_df[
        trial_predictions_df["Run"] == baseline_run
    ][prediction_columns].copy()
    baseline_trials["Trial_Euclidean_Error"] = np.hypot(
        baseline_trials["Regression_Pred_X"] - baseline_trials["True_X"],
        baseline_trials["Regression_Pred_Y"] - baseline_trials["True_Y"],
    )
    probe_errors = baseline_trials.groupby(keys, as_index=False).agg(
        Fixation_Probe_Avg_Euclidean_Error=("Trial_Euclidean_Error", "mean"),
        Fixation_Probe_Num_Trials=("Trial_Euclidean_Error", "size"),
    )
    available_keys = available_probe_keys[keys].drop_duplicates()
    probe_errors = probe_errors.merge(
        available_keys, on=keys, how="inner", validate="one_to_one"
    )
    probe_errors = probe_errors[
        np.isfinite(probe_errors["Fixation_Probe_Avg_Euclidean_Error"])
    ].copy()
    if probe_errors.empty:
        raise ValueError("No finite fixation probe errors are available for raw ranking.")
    session_keys = [column for column in keys if column not in ("True_X", "True_Y")]
    tie_break_columns = ["Fixation_Probe_Avg_Euclidean_Error"] + session_keys + ["True_X", "True_Y"]
    probe_errors = probe_errors.sort_values(tie_break_columns, kind="stable").reset_index(drop=True)
    probe_errors["Raw_Probe_Rank"] = np.arange(1, len(probe_errors) + 1)
    return probe_errors


def save_empty_aware_unaware_scatter_plots(
        scope_title, scope_suffix, root_result_dir, point_level, axes=("X", "Y"),
        file_extensions=("png",)):
    """Save explicit placeholder scatter plots when a filtered scope has no points."""
    os.makedirs(root_result_dir, exist_ok=True)
    suffix = f"_{scope_suffix}" if scope_suffix else ""
    for axis in axes:
        fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
        ax.text(0.5, 0.5, "No eligible data after min-3/session filter",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_xlabel(f"Unaware Signed Delta {axis}")
        ax.set_ylabel(f"Aware Signed Delta {axis}")
        ax.set_title(f"Aware vs Unaware Signed Mislocalization (Delta {axis})\n{scope_title}")
        ax.text(0.03, 0.97, f"n=0; {point_level}", transform=ax.transAxes, va="top")
        ax.grid(True, alpha=0.25)
        output_stem = os.path.join(
            root_result_dir,
            f"aware_vs_unaware_delta_{axis.lower()}_signed_scatter{suffix}",
        )
        for file_extension in file_extensions:
            fig.savefig(f"{output_stem}.{file_extension}")
        plt.close(fig)


def plot_aware_unaware_signed_error_distributions(
        selected_df, scope_title, scope_suffix, root_result_dir,
        point_level, axes=("X", "Y"), statistics_df=None):
    """Save separate paired signed-error distribution figures for requested axes."""
    os.makedirs(root_result_dir, exist_ok=True)
    state_colors = ["tab:blue", "tab:orange"]  # Unaware, Aware
    rng = np.random.default_rng(42)
    suffix = f"_{scope_suffix}" if scope_suffix else ""
    for axis in axes:
        direction_text = (
            "Opposite saccade direction (-) / Saccade direction (+)"
            if axis == "X" else "Down (-) / Up (+)"
        )
        unaware = selected_df[f"Unaware_Delta_{axis}"].to_numpy(dtype=float)
        aware = selected_df[f"Aware_Delta_{axis}"].to_numpy(dtype=float)
        finite = np.isfinite(unaware) & np.isfinite(aware)
        unaware, aware = unaware[finite], aware[finite]
        if len(unaware) == 0:
            fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
            ax.text(0.5, 0.5, "No eligible data after min-3/session filter",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_xticks([1, 2], ["Unaware", "Aware"])
            ax.set_ylabel(f"Signed Delta {axis} from fixation baseline (grid units)")
            ax.set_title(f"Signed Error Distribution: Delta {axis}\n{direction_text}")
            fig.suptitle(f"Aware vs Unaware Signed Error\n{scope_title}; paired {point_level}; n=0")
            fig.savefig(os.path.join(
                root_result_dir,
                f"aware_vs_unaware_signed_error_distribution_delta_{axis.lower()}{suffix}.png",
            ), dpi=200)
            plt.close(fig)
            continue

        fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
        values = [unaware, aware]
        for position, data, color in zip((1, 2), values, state_colors):
            if len(data) > 1 and np.ptp(data) > 0:
                violin = ax.violinplot(
                    [data], positions=[position], widths=0.72,
                    showmeans=False, showmedians=False, showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.24)
        box = ax.boxplot(
            values, positions=[1, 2], widths=0.28, patch_artist=True,
            showfliers=False, medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch_box, color in zip(box["boxes"], state_colors):
            patch_box.set_facecolor(color)
            patch_box.set_alpha(0.45)

        jitter = rng.uniform(-0.07, 0.07, size=len(unaware))
        line_alpha = 0.08 if len(unaware) > 100 else 0.20
        point_alpha = 0.14 if len(unaware) > 100 else 0.40
        for index in range(len(unaware)):
            ax.plot(
                [1 + jitter[index], 2 + jitter[index]], [unaware[index], aware[index]],
                color="0.45", linewidth=0.5, alpha=line_alpha, zorder=1,
            )
        ax.scatter(1 + jitter, unaware, s=10, color=state_colors[0], alpha=point_alpha, zorder=2)
        ax.scatter(2 + jitter, aware, s=10, color=state_colors[1], alpha=point_alpha, zorder=2)
        ax.set_xticks([1, 2], ["Unaware", "Aware"])
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        plot_limit = max(float(np.max(np.abs(np.concatenate(values)))) * 1.10, 1e-6)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_ylabel(f"Signed Δ{axis} from fixation baseline (grid units)")
        ax.set_title(f"Signed Error Distribution: Δ{axis}\n{direction_text}")
        ax.grid(axis="y", alpha=0.25)
        medians = np.median(unaware), np.median(aware)
        ax.text(
            0.03, 0.97,
            f"n={len(unaware)}\nmedian: {medians[0]:.3f} → {medians[1]:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
        )
        if statistics_df is not None:
            stats_row = statistics_df[
                (statistics_df["Scope"] == (scope_suffix or "all_probes"))
                & (statistics_df["Axis"] == axis)
                & (statistics_df["Measure"] == "Signed_Delta")
            ].iloc[0]
            unaware_p = float(stats_row["Unaware_Bias_Wilcoxon_P"])
            aware_p = float(stats_row["Aware_Bias_Wilcoxon_P"])
            paired_p = float(stats_row["Wilcoxon_P"])
            ax.set_xticks([1, 2], [
                f"Unaware\nvs 0: {significance_stars(unaware_p)} "
                f"(p={format_probability(unaware_p)})",
                f"Aware\nvs 0: {significance_stars(aware_p)} "
                f"(p={format_probability(aware_p)})",
            ])
            bracket_y = plot_limit * 1.06
            bracket_height = plot_limit * 0.05
            ax.plot(
                [1, 1, 2, 2],
                [bracket_y, bracket_y + bracket_height,
                 bracket_y + bracket_height, bracket_y],
                color="black", linewidth=1.0, clip_on=False,
            )
            ax.text(
                1.5, bracket_y + bracket_height,
                f"{significance_stars(paired_p)}  paired p={format_probability(paired_p)}",
                ha="center", va="bottom", fontsize=9,
            )
            ax.set_ylim(-plot_limit * 1.25, plot_limit * 1.25)
        fig.suptitle(
            f"Aware vs Unaware Signed Error\n{scope_title}; paired {point_level}"
        )
        fig.savefig(
            os.path.join(
                root_result_dir,
                f"aware_vs_unaware_signed_error_distribution_delta_{axis.lower()}{suffix}.png",
            ),
            dpi=200,
        )
        plt.close(fig)


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg FDR-adjusted p-values, preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if len(finite_indices) == 0:
        return adjusted
    ordered_indices = finite_indices[np.argsort(p_values[finite_indices], kind="stable")]
    ordered_p = p_values[ordered_indices]
    num_tests = len(ordered_p)
    ordered_adjusted = ordered_p * num_tests / np.arange(1, num_tests + 1)
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    adjusted[ordered_indices] = np.clip(ordered_adjusted, 0.0, 1.0)
    return adjusted


def rank_biserial(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum())


def one_sample_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if np.count_nonzero(values) == 0:
        return 0.0, 1.0
    result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def calculate_aware_unaware_statistics(
        scopes, session_key_columns, bootstrap_samples=5000):
    """Run condition-bias and paired tests after aggregation to independent sessions."""
    rows = []
    rng = np.random.default_rng(20260720)
    unit_label = " + ".join(session_key_columns)
    for scope_suffix, selected_df, scope_title in scopes:
        for axis in ("X", "Y"):
            for measure, unaware_column, aware_column in (
                    ("Signed_Delta", f"Unaware_Delta_{axis}", f"Aware_Delta_{axis}"),
                    ("Absolute_Delta", f"Unaware_Abs_Delta_{axis}", f"Aware_Abs_Delta_{axis}")):
                finite_rows = selected_df[
                    np.isfinite(selected_df[unaware_column])
                    & np.isfinite(selected_df[aware_column])
                ]
                grouped = finite_rows.groupby(session_key_columns, as_index=False)[
                    [unaware_column, aware_column]
                ].mean()
                unaware = grouped[unaware_column].to_numpy(dtype=float)
                aware = grouped[aware_column].to_numpy(dtype=float)
                differences = aware - unaware
                n_units = len(differences)

                if n_units:
                    mean_difference = float(np.mean(differences))
                    median_difference = float(np.median(differences))
                    if n_units > 1:
                        indices = rng.integers(0, n_units, size=(bootstrap_samples, n_units))
                        diff_bootstrap = differences[indices].mean(axis=1)
                        diff_ci_low, diff_ci_high = np.quantile(diff_bootstrap, [0.025, 0.975])
                        difference_sd = float(np.std(differences, ddof=1))
                        if difference_sd > np.finfo(float).eps * max(1.0, float(np.max(np.abs(differences)))):
                            cohen_dz = mean_difference / difference_sd
                            t_result = ttest_rel(aware, unaware, nan_policy="omit")
                            t_statistic, t_p = float(t_result.statistic), float(t_result.pvalue)
                        else:
                            cohen_dz = t_statistic = t_p = np.nan
                    else:
                        indices = None
                        diff_ci_low = diff_ci_high = mean_difference
                        cohen_dz = t_statistic = t_p = np.nan
                    paired_statistic, paired_p = one_sample_wilcoxon(differences)
                    shapiro_p = (
                        float(shapiro(differences).pvalue)
                        if 3 <= n_units <= 5000 and np.ptp(differences) > np.finfo(float).eps * max(1.0, float(np.max(np.abs(differences)))) else np.nan
                    )
                    if measure == "Signed_Delta":
                        unaware_statistic, unaware_p = one_sample_wilcoxon(unaware)
                        aware_statistic, aware_p = one_sample_wilcoxon(aware)
                        if indices is not None:
                            unaware_ci_low, unaware_ci_high = np.quantile(
                                unaware[indices].mean(axis=1), [0.025, 0.975]
                            )
                            aware_ci_low, aware_ci_high = np.quantile(
                                aware[indices].mean(axis=1), [0.025, 0.975]
                            )
                        else:
                            unaware_ci_low = unaware_ci_high = unaware[0]
                            aware_ci_low = aware_ci_high = aware[0]
                    else:
                        unaware_statistic = unaware_p = aware_statistic = aware_p = np.nan
                        unaware_ci_low = unaware_ci_high = aware_ci_low = aware_ci_high = np.nan
                else:
                    mean_difference = median_difference = diff_ci_low = diff_ci_high = np.nan
                    cohen_dz = t_statistic = t_p = shapiro_p = np.nan
                    paired_statistic = paired_p = np.nan
                    unaware_statistic = unaware_p = aware_statistic = aware_p = np.nan
                    unaware_ci_low = unaware_ci_high = aware_ci_low = aware_ci_high = np.nan

                rows.append({
                    "Scope": scope_suffix or "all_probes", "Scope_Title": scope_title,
                    "Axis": axis, "Measure": measure, "Independent_Unit": unit_label,
                    "N_Independent_Units": n_units,
                    "N_Session_Probe_or_Trial_Pairs": len(finite_rows),
                    "Unaware_Session_Mean": float(np.mean(unaware)) if n_units else np.nan,
                    "Aware_Session_Mean": float(np.mean(aware)) if n_units else np.nan,
                    "Unaware_Session_Median": float(np.median(unaware)) if n_units else np.nan,
                    "Aware_Session_Median": float(np.median(aware)) if n_units else np.nan,
                    "Unaware_Mean_Bias_CI95_Low": float(unaware_ci_low),
                    "Unaware_Mean_Bias_CI95_High": float(unaware_ci_high),
                    "Aware_Mean_Bias_CI95_Low": float(aware_ci_low),
                    "Aware_Mean_Bias_CI95_High": float(aware_ci_high),
                    "Unaware_Bias_Wilcoxon_Statistic": unaware_statistic,
                    "Unaware_Bias_Wilcoxon_P": unaware_p,
                    "Unaware_Bias_Rank_Biserial": rank_biserial(unaware),
                    "Aware_Bias_Wilcoxon_Statistic": aware_statistic,
                    "Aware_Bias_Wilcoxon_P": aware_p,
                    "Aware_Bias_Rank_Biserial": rank_biserial(aware),
                    "Mean_Difference_Aware_Minus_Unaware": mean_difference,
                    "Median_Difference_Aware_Minus_Unaware": median_difference,
                    "Mean_Difference_CI95_Low": float(diff_ci_low),
                    "Mean_Difference_CI95_High": float(diff_ci_high),
                    "Cohen_dz": cohen_dz,
                    "Rank_Biserial_Correlation": rank_biserial(differences),
                    "Wilcoxon_Statistic": paired_statistic, "Wilcoxon_P": paired_p,
                    "Paired_T_Statistic": t_statistic, "Paired_T_P": t_p,
                    "Shapiro_Difference_P": shapiro_p,
                })

    statistics_df = pd.DataFrame(rows)
    statistics_df["Wilcoxon_FDR_Q"] = benjamini_hochberg(statistics_df["Wilcoxon_P"])
    statistics_df["Wilcoxon_FDR_Significant_0_05"] = statistics_df["Wilcoxon_FDR_Q"] < 0.05
    signed_mask = statistics_df["Measure"] == "Signed_Delta"
    for condition in ("Unaware", "Aware"):
        p_column = f"{condition}_Bias_Wilcoxon_P"
        q_column = f"{condition}_Bias_FDR_Q"
        statistics_df[q_column] = np.nan
        statistics_df.loc[signed_mask, q_column] = benjamini_hochberg(
            statistics_df.loc[signed_mask, p_column]
        )
        statistics_df[f"{condition}_Bias_FDR_Significant_0_05"] = (
            statistics_df[q_column] < 0.05
        )
    return statistics_df


def format_probability(value):
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def significance_stars(p_value):
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def discover_cross_sessions(base_data_dir, requested_sessions=None, excluded_sessions=None):
    excluded = set(excluded_sessions or [])
    if requested_sessions:
        candidates = requested_sessions
    else:
        candidates = sorted(entry for entry in os.listdir(base_data_dir)
                            if os.path.isdir(os.path.join(base_data_dir, entry)))
    return [session for session in candidates if session not in excluded
            and all(os.path.isdir(os.path.join(base_data_dir, session, cfg[0]))
                    for cfg in CONDITION_CONFIG.values())]


def find_condition_units(base_data_dir, session, condition, max_channel_id=16,
                         excluded_channel_ids=(17,)):
    subdir, suffix, _, _ = CONDITION_CONFIG[condition]
    condition_dir = os.path.join(base_data_dir, session, subdir)
    pattern = re.compile(
        rf"^spike_probe_{re.escape(str(session))}_(\d+)_(\d+)_{suffix}\.mat$"
    )
    excluded = set(excluded_channel_ids)
    units = {}
    for filename in os.listdir(condition_dir):
        match = pattern.match(filename)
        if not match:
            continue
        channel_id, unit_id = map(int, match.groups())
        if channel_id <= max_channel_id and channel_id not in excluded:
            units[(channel_id, unit_id)] = os.path.join(condition_dir, filename)
    return units


def common_condition_units(base_data_dir, session):
    fixation = find_condition_units(base_data_dir, session, "fixation")
    peri = find_condition_units(base_data_dir, session, "peri_saccade")
    common = sorted(set(fixation) & set(peri))
    if not common:
        raise RuntimeError(f"No common fixation/peri-saccade units in session {session}.")
    return common, fixation, peri


def _sdf(spikes, kernel_std_samples):
    size = max(3, int(kernel_std_samples * 6))
    size += size % 2 == 0
    kernel = signal.windows.gaussian(size, std=kernel_std_samples, sym=True)
    kernel /= kernel.sum()
    return np.apply_along_axis(
        lambda row: signal.convolve(row, kernel, mode="same"), 1,
        np.asarray(spikes, dtype=np.float32),
    )


def load_condition_features(file_map, common_units, condition, mapping, time_window,
                            bin_size_samples, kernel_std_samples, horizontal=False,
                            selected_condition_labels=None):
    _, _, spike_key, label_key = CONDITION_CONFIG[condition]
    features = []
    reference_labels = None
    window_length = time_window.stop - time_window.start
    if window_length % bin_size_samples:
        raise ValueError("SDF time-window length must be divisible by bin size.")
    for unit in common_units:
        mat = scipy.io.loadmat(file_map[unit])
        spikes = np.asarray(mat[spike_key])
        labels = np.squeeze(np.asarray(mat[label_key]))
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise ValueError(f"Condition labels differ between units for {condition}.")
        if spikes.shape[0] != len(labels):
            raise ValueError(f"Trial count mismatch for {condition}, unit {unit}.")
        sdf_window = _sdf(spikes, kernel_std_samples)[:, time_window]
        n_bins = window_length // bin_size_samples
        features.append(sdf_window.reshape(len(labels), n_bins, bin_size_samples).mean(axis=2))
    X = np.stack(features, axis=1).reshape(len(reference_labels), -1).astype(np.float32)
    if selected_condition_labels is not None:
        selected_condition_labels = tuple(int(value) for value in selected_condition_labels)
        selected_mask = np.isin(reference_labels, selected_condition_labels)
        X = X[selected_mask]
        reference_labels = reference_labels[selected_mask]
        missing = sorted(set(selected_condition_labels) - set(map(int, reference_labels)))
        if missing:
            raise ValueError(
                f"{condition} is missing selected probe condition label(s): {missing}."
            )
    y2d = np.asarray([mapping[int(label)] for label in reference_labels], dtype=np.float32)
    return X, (y2d[:, 0] if horizontal else y2d), reference_labels


def apply_grid_projection(predictions, weight):
    predictions = np.asarray(predictions)
    nearest = np.clip(np.rint(predictions), 0.0, 2.0)
    return predictions + weight * (nearest - predictions)


def _balanced_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim == 1:
        errors = [np.mean(np.abs(y_pred[y_true == loc] - loc)) for loc in np.unique(y_true)]
    else:
        errors = [np.mean(np.linalg.norm(y_pred[np.all(y_true == loc, axis=1)] - loc, axis=1))
                  for loc in np.unique(y_true, axis=0)]
    return float(np.mean(errors))


def _model(alpha, args):
    length_scale_bounds = getattr(args, "length_scale_bounds", (0.1, 10.0))
    kernel = ConstantKernel(1.0) * RBF(
        length_scale=args.length_scale, length_scale_bounds=length_scale_bounds
    )
    return GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state,
    )


def tune_condition_loo(X, y, args, run_name, summarize):
    if len(X) < 2:
        raise ValueError(f"{run_name} needs at least two trials.")
    max_components = min(len(X) - 1, X.shape[1])
    if args.pca_components > max_components:
        raise ValueError(
            f"{run_name}: PCA components={args.pca_components}, maximum in LOO={max_components}."
        )
    folds = []
    for train_index, val_index in LeaveOneOut().split(X):
        pca = PCA(n_components=args.pca_components, random_state=args.random_state)
        folds.append((train_index, val_index, pca.fit_transform(X[train_index]), pca.transform(X[val_index])))

    tuning_rows, best = [], None
    for alpha in args.alpha_candidates:
        raw_oof = np.empty_like(y, dtype=float)
        raw_train, kernels = [], []
        for train_index, val_index, train_pca, val_pca in folds:
            model = _model(alpha, args).fit(train_pca, y[train_index])
            raw_train.append(model.predict(train_pca))
            raw_oof[val_index] = model.predict(val_pca)
            kernels.append(str(model.kernel_))
        for weight in args.grid_projection_weights:
            oof = apply_grid_projection(raw_oof, weight)
            balanced = _balanced_error(y, oof)
            mse = float(mean_squared_error(y, oof))
            row = {"Run": run_name, "Alpha": alpha, "Grid_Projection_Weight": weight,
                   "LOO_MSE": mse, "Probe_Balanced_Error": balanced}
            tuning_rows.append(row)
            key = (balanced, mse)
            if best is None or key < best["key"]:
                best = {"key": key, "alpha": alpha, "weight": weight, "raw_oof": raw_oof.copy(),
                        "oof": oof.copy(), "raw_train": [p.copy() for p in raw_train], "kernels": kernels[:]}

    fold_rows, prediction_rows, train_losses, val_losses = [], [], [], []
    horizontal = np.asarray(y).ndim == 1
    for fold_no, ((train_index, val_index, _, _), train_raw, kernel) in enumerate(
            zip(folds, best["raw_train"], best["kernels"]), 1):
        train_pred = apply_grid_projection(train_raw, best["weight"])
        train_mse = float(mean_squared_error(y[train_index], train_pred))
        val_mse = float(mean_squared_error(y[val_index], best["oof"][val_index]))
        train_losses.append(train_mse); val_losses.append(val_mse)
        idx = int(val_index[0])
        row = {"Run": run_name, "Fold": fold_no, "LOO_Test_Index": idx,
               "Train_MSE": train_mse, "Validation_MSE": val_mse,
               "Alpha": best["alpha"], "Grid_Projection_Weight": best["weight"],
               "Learned_Kernel": kernel}
        if horizontal:
            row.update(True_X=float(y[idx]), Raw_Pred_X=float(best["raw_oof"][idx]),
                       Regression_Pred_X=float(best["oof"][idx]))
        else:
            row.update(True_X=float(y[idx, 0]), True_Y=float(y[idx, 1]),
                       Raw_Pred_X=float(best["raw_oof"][idx, 0]), Raw_Pred_Y=float(best["raw_oof"][idx, 1]),
                       Regression_Pred_X=float(best["oof"][idx, 0]), Regression_Pred_Y=float(best["oof"][idx, 1]))
        fold_rows.append(row)
        prediction_rows.append({key: value for key, value in row.items()
                                if key not in {"Fold", "Train_MSE", "Validation_MSE", "Learned_Kernel"}})

    metrics = summarize(y, best["oof"], run_name)
    metrics.update(Mean_Train_MSE=float(np.mean(train_losses)),
                   Mean_Validation_MSE=float(np.mean(val_losses)), CV="leave_one_out_tuned",
                   Selected_Alpha=best["alpha"], Selected_Grid_Projection_Weight=best["weight"],
                   PCA_Components=args.pca_components)
    return {"name": run_name, "y_true": y, "y_pred": best["oof"], "metrics": metrics,
            "fold_df": pd.DataFrame(fold_rows), "prediction_df": pd.DataFrame(prediction_rows),
            "tuning_df": pd.DataFrame(tuning_rows), "alpha": best["alpha"], "weight": best["weight"]}


def cross_condition_decode(trainX, trainY, testX, testY, tuned_run, args, run_name, summarize):
    if trainX.shape[1] != testX.shape[1]:
        raise ValueError("Train/test feature dimensions differ; common units and bins are required.")
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    train_pca = pca.fit_transform(trainX)
    test_pca = pca.transform(testX)
    model = _model(tuned_run["alpha"], args).fit(train_pca, trainY)
    raw_train = model.predict(train_pca)
    raw_test = model.predict(test_pca)
    train_pred = apply_grid_projection(raw_train, tuned_run["weight"])
    test_pred = apply_grid_projection(raw_test, tuned_run["weight"])
    metrics = summarize(testY, test_pred, run_name)
    metrics.update(Train_MSE=float(mean_squared_error(trainY, train_pred)),
                   CV="train_full_condition_test_other_condition",
                   Selected_Alpha=tuned_run["alpha"],
                   Selected_Grid_Projection_Weight=tuned_run["weight"],
                   PCA_Components=args.pca_components, Learned_Kernel=str(model.kernel_))
    horizontal = np.asarray(testY).ndim == 1
    data = {"Run": run_name, "Test_Index": np.arange(len(testY))}
    if horizontal:
        data.update(True_X=testY, Raw_Pred_X=raw_test, Regression_Pred_X=test_pred)
    else:
        data.update(True_X=testY[:, 0], True_Y=testY[:, 1], Raw_Pred_X=raw_test[:, 0],
                    Raw_Pred_Y=raw_test[:, 1], Regression_Pred_X=test_pred[:, 0],
                    Regression_Pred_Y=test_pred[:, 1])
    return {"name": run_name, "y_true": testY, "y_pred": test_pred,
            "metrics": metrics, "prediction_df": pd.DataFrame(data)}


def tune_condition_loo_no_grid(X, y, args, run_name, summarize):
    """Tune alpha with configurable CV and fold-local scaling/PCA."""
    if len(X) < 2:
        raise ValueError(f"{run_name} needs at least two trials.")
    requested_folds = getattr(args, "tuning_cv_folds", None)
    if requested_folds is None:
        splitter = LeaveOneOut()
        cv_name = "leave_one_out"
        cv_folds = len(X)
    else:
        cv_folds = int(requested_folds)
        if cv_folds < 2 or cv_folds > len(X):
            raise ValueError(
                f"{run_name}: CV folds must be between 2 and {len(X)}, got {cv_folds}."
            )
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)
        cv_name = f"{cv_folds}_fold"
    split_indices = list(splitter.split(X))
    min_train_samples = min(len(train_index) for train_index, _ in split_indices)
    max_components = min(min_train_samples, X.shape[1])
    if args.pca_components > max_components:
        raise ValueError(
            f"{run_name}: PCA components={args.pca_components}, "
            f"maximum in {cv_name} CV={max_components}."
        )
    folds = []
    for train_index, val_index in split_indices:
        train_features, val_features = X[train_index], X[val_index]
        if getattr(args, "standardize_features", False):
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)
        pca = PCA(n_components=args.pca_components, random_state=args.random_state)
        folds.append((train_index, val_index,
                      pca.fit_transform(train_features), pca.transform(val_features)))

    tuning_rows, best = [], None
    for alpha in args.alpha_candidates:
        oof = np.empty_like(y, dtype=float)
        train_predictions, kernels = [], []
        for train_index, val_index, train_pca, val_pca in folds:
            model = _model(alpha, args).fit(train_pca, y[train_index])
            train_predictions.append(model.predict(train_pca))
            oof[val_index] = model.predict(val_pca)
            kernels.append(str(model.kernel_))
        balanced = _balanced_error(y, oof)
        mse = float(mean_squared_error(y, oof))
        tuning_rows.append({
            "Run": run_name, "Alpha": alpha, "LOO_MSE": mse,
            "Probe_Balanced_Error": balanced, "CV_Method": cv_name, "CV_Folds": cv_folds,
        })
        key = (balanced, mse)
        if best is None or key < best["key"]:
            best = {"key": key, "alpha": alpha, "oof": oof.copy(),
                    "train_predictions": [prediction.copy() for prediction in train_predictions],
                    "kernels": kernels[:]}

    horizontal = np.asarray(y).ndim == 1
    fold_rows, prediction_rows, train_losses, val_losses = [], [], [], []
    for fold_no, ((train_index, val_index, _, _), train_pred, kernel) in enumerate(
            zip(folds, best["train_predictions"], best["kernels"]), 1):
        train_mse = float(mean_squared_error(y[train_index], train_pred))
        val_mse = float(mean_squared_error(y[val_index], best["oof"][val_index]))
        train_losses.append(train_mse); val_losses.append(val_mse)
        for validation_index in val_index:
            idx = int(validation_index)
            row = {"Run": run_name, "Fold": fold_no, "LOO_Test_Index": idx,
                   "Train_MSE": train_mse, "Validation_MSE": val_mse,
                   "Selected_Alpha": best["alpha"], "Learned_Kernel": kernel,
                   "CV_Method": cv_name, "CV_Folds": cv_folds}
            if horizontal:
                row.update(True_X=float(y[idx]), Regression_Pred_X=float(best["oof"][idx]))
            else:
                row.update(True_X=float(y[idx, 0]), True_Y=float(y[idx, 1]),
                           Regression_Pred_X=float(best["oof"][idx, 0]),
                           Regression_Pred_Y=float(best["oof"][idx, 1]))
            fold_rows.append(row)
            prediction_rows.append({key: value for key, value in row.items()
                                    if key not in {"Fold", "Train_MSE", "Validation_MSE",
                                                   "Learned_Kernel"}})

    metrics = summarize(y, best["oof"], run_name)
    metrics.update(Mean_Train_MSE=float(np.mean(train_losses)),
                   Mean_Validation_MSE=float(np.mean(val_losses)),
                   CV=f"{cv_name}_alpha_tuned_no_grid", CV_Folds=cv_folds,
                   Selected_Alpha=best["alpha"],
                   PCA_Components=args.pca_components, Grid_Projection=False)
    return {"name": run_name, "y_true": y, "y_pred": best["oof"], "metrics": metrics,
            "fold_df": pd.DataFrame(fold_rows), "prediction_df": pd.DataFrame(prediction_rows),
            "tuning_df": pd.DataFrame(tuning_rows), "alpha": best["alpha"]}


def cross_condition_decode_no_grid(trainX, trainY, testX, testY, tuned_run, args,
                                   run_name, summarize):
    """Fit source-condition PCA/GPR and return raw target-condition predictions."""
    if trainX.shape[1] != testX.shape[1]:
        raise ValueError("Train/test feature dimensions differ; common units and bins are required.")
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    train_pca = pca.fit_transform(trainX)
    test_pca = pca.transform(testX)
    model = _model(tuned_run["alpha"], args).fit(train_pca, trainY)
    train_pred = model.predict(train_pca)
    test_pred = model.predict(test_pca)
    metrics = summarize(testY, test_pred, run_name)
    metrics.update(Train_MSE=float(mean_squared_error(trainY, train_pred)),
                   CV="train_full_condition_test_other_condition_no_grid",
                   Selected_Alpha=tuned_run["alpha"], PCA_Components=args.pca_components,
                   Grid_Projection=False, Learned_Kernel=str(model.kernel_))
    horizontal = np.asarray(testY).ndim == 1
    data = {"Run": run_name, "Test_Index": np.arange(len(testY)),
            "Selected_Alpha": tuned_run["alpha"]}
    if horizontal:
        data.update(True_X=testY, Regression_Pred_X=test_pred)
    else:
        data.update(True_X=testY[:, 0], True_Y=testY[:, 1],
                    Regression_Pred_X=test_pred[:, 0], Regression_Pred_Y=test_pred[:, 1])
    return {"name": run_name, "y_true": testY, "y_pred": test_pred,
            "metrics": metrics, "prediction_df": pd.DataFrame(data)}


def tune_condition_loo_independent_xy_no_grid(X, y, args, run_name, summarize, fixed_alphas=None):
    """Tune and decode X/Y with two independent scalar GPR models."""
    if np.asarray(y).ndim != 2 or y.shape[1] != 2:
        raise ValueError("Independent X/Y decoding requires y with shape (n_samples, 2).")

    def axis_summary(y_true, y_pred, name):
        return {"Run": name, "MSE": float(mean_squared_error(y_true, y_pred))}

    axis_results = []
    for axis_index, axis_name in enumerate(("X", "Y")):
        axis_args = args
        if fixed_alphas is not None:
            axis_args = copy.copy(args)
            axis_args.alpha_candidates = [float(fixed_alphas[axis_index])]
        result = tune_condition_loo_no_grid(
            X, y[:, axis_index], axis_args, f"{run_name}_{axis_name}", axis_summary
        )
        result["axis"] = axis_name
        axis_results.append(result)
    x_result, y_result = axis_results
    predictions = np.column_stack([x_result["y_pred"], y_result["y_pred"]])

    x_folds = x_result["fold_df"].rename(columns={
        "Train_MSE": "Train_MSE_X", "Validation_MSE": "Validation_MSE_X",
        "Selected_Alpha": "Selected_Alpha_X", "Learned_Kernel": "Learned_Kernel_X",
    })
    y_folds = y_result["fold_df"].rename(columns={
        "Train_MSE": "Train_MSE_Y", "Validation_MSE": "Validation_MSE_Y",
        "Selected_Alpha": "Selected_Alpha_Y", "Learned_Kernel": "Learned_Kernel_Y",
    })
    fold_df = x_folds[["Fold", "LOO_Test_Index", "Train_MSE_X", "Validation_MSE_X",
                        "Selected_Alpha_X", "Learned_Kernel_X", "CV_Method", "CV_Folds"]].merge(
        y_folds[["Fold", "LOO_Test_Index", "Train_MSE_Y", "Validation_MSE_Y",
                 "Selected_Alpha_Y", "Learned_Kernel_Y"]],
        on=["Fold", "LOO_Test_Index"], how="inner",
    )
    fold_df.insert(0, "Run", run_name)
    prediction_df = pd.DataFrame({
        "Run": run_name, "LOO_Test_Index": np.arange(len(y)),
        "Selected_Alpha_X": x_result["alpha"], "Selected_Alpha_Y": y_result["alpha"],
        "True_X": y[:, 0], "True_Y": y[:, 1],
        "Regression_Pred_X": predictions[:, 0], "Regression_Pred_Y": predictions[:, 1],
    })
    tuning_df = pd.concat([
        x_result["tuning_df"].assign(Run=run_name, Axis="X"),
        y_result["tuning_df"].assign(Run=run_name, Axis="Y"),
    ], ignore_index=True)
    metrics = summarize(y, predictions, run_name)
    metrics.update(
        Selected_Alpha_X=x_result["alpha"], Selected_Alpha_Y=y_result["alpha"],
        X_LOO_MSE=float(mean_squared_error(y[:, 0], predictions[:, 0])),
        Y_LOO_MSE=float(mean_squared_error(y[:, 1], predictions[:, 1])),
        CV=(f"{int(args.tuning_cv_folds)}_fold_independent_xy_alpha_tuned_no_grid"
            if getattr(args, "tuning_cv_folds", None) is not None
            else "leave_one_out_independent_xy_alpha_tuned_no_grid"),
        PCA_Components=args.pca_components, Grid_Projection=False,
    )
    return {"name": run_name, "y_true": y, "y_pred": predictions, "metrics": metrics,
            "fold_df": fold_df, "prediction_df": prediction_df, "tuning_df": tuning_df,
            "alpha_x": x_result["alpha"], "alpha_y": y_result["alpha"]}


def cross_condition_decode_independent_xy_no_grid(trainX, trainY, testX, testY,
                                                   tuned_run, args, run_name, summarize):
    """Cross-condition prediction with separate scalar X and Y GPR models."""
    if trainX.shape[1] != testX.shape[1]:
        raise ValueError("Train/test feature dimensions differ; common units and bins are required.")
    train_features, test_features = trainX, testX
    if getattr(args, "standardize_features", False):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)
    train_predictions, test_predictions, kernels = [], [], []
    for axis_index, alpha in enumerate((tuned_run["alpha_x"], tuned_run["alpha_y"])):
        model = _model(alpha, args).fit(train_pca, trainY[:, axis_index])
        train_predictions.append(model.predict(train_pca))
        test_predictions.append(model.predict(test_pca))
        kernels.append(str(model.kernel_))
    train_pred = np.column_stack(train_predictions)
    test_pred = np.column_stack(test_predictions)
    metrics = summarize(testY, test_pred, run_name)
    metrics.update(
        Train_MSE=float(mean_squared_error(trainY, train_pred)),
        X_Test_MSE=float(mean_squared_error(testY[:, 0], test_pred[:, 0])),
        Y_Test_MSE=float(mean_squared_error(testY[:, 1], test_pred[:, 1])),
        CV="train_full_condition_test_other_condition_independent_xy_no_grid",
        Selected_Alpha_X=tuned_run["alpha_x"], Selected_Alpha_Y=tuned_run["alpha_y"],
        PCA_Components=args.pca_components, Grid_Projection=False,
        Learned_Kernel_X=kernels[0], Learned_Kernel_Y=kernels[1],
    )
    prediction_df = pd.DataFrame({
        "Run": run_name, "Test_Index": np.arange(len(testY)),
        "Selected_Alpha_X": tuned_run["alpha_x"], "Selected_Alpha_Y": tuned_run["alpha_y"],
        "True_X": testY[:, 0], "True_Y": testY[:, 1],
        "Regression_Pred_X": test_pred[:, 0], "Regression_Pred_Y": test_pred[:, 1],
    })
    return {"name": run_name, "y_true": testY, "y_pred": test_pred,
            "metrics": metrics, "prediction_df": prediction_df}


def tune_conditions_shared_independent_xy_no_grid(fixX, fixY, periX, periY, args,
                                                   fix_run_name, peri_run_name, summarize):
    """Choose one alpha per axis from equal-weighted fixation/peri CV performance."""
    tuning_cv_folds = getattr(args, "tuning_cv_folds", None)
    tuning_cv_method = (
        f"{int(tuning_cv_folds)}_fold" if tuning_cv_folds is not None else "leave_one_out"
    )
    candidate_results = {}
    tuning_rows = []
    for alpha in args.alpha_candidates:
        candidate_args = copy.copy(args)
        candidate_args.alpha_candidates = [alpha]
        fix_result = tune_condition_loo_independent_xy_no_grid(
            fixX, fixY, candidate_args, fix_run_name, summarize
        )
        peri_result = tune_condition_loo_independent_xy_no_grid(
            periX, periY, candidate_args, peri_run_name, summarize
        )
        candidate_results[alpha] = {"fixation": fix_result, "peri_saccade": peri_result}
        for axis, mse_key in (("X", "X_LOO_MSE"), ("Y", "Y_LOO_MSE")):
            fix_probe_error = float(fix_result["tuning_df"].loc[
                fix_result["tuning_df"]["Axis"] == axis, "Probe_Balanced_Error"
            ].iloc[0])
            peri_probe_error = float(peri_result["tuning_df"].loc[
                peri_result["tuning_df"]["Axis"] == axis, "Probe_Balanced_Error"
            ].iloc[0])
            fix_mse = float(fix_result["metrics"][mse_key])
            peri_mse = float(peri_result["metrics"][mse_key])
            tuning_rows.append({
                "Axis": axis,
                "Alpha": alpha,
                "CV_Method": tuning_cv_method,
                "CV_Folds": int(tuning_cv_folds) if tuning_cv_folds is not None else len(fixX),
                "Fixation_Probe_Balanced_Error": fix_probe_error,
                "Peri_Saccade_Probe_Balanced_Error": peri_probe_error,
                "Combined_Probe_Balanced_Error": (fix_probe_error + peri_probe_error) / 2.0,
                "Fixation_CV_MSE": fix_mse,
                "Peri_Saccade_CV_MSE": peri_mse,
                "Combined_CV_MSE": (fix_mse + peri_mse) / 2.0,
                "Fixation_LOO_MSE": fix_mse,
                "Peri_Saccade_LOO_MSE": peri_mse,
                "Combined_LOO_MSE": (fix_mse + peri_mse) / 2.0,
            })

    tuning_df = pd.DataFrame(tuning_rows)
    selected_alphas = {}
    for axis in ("X", "Y"):
        axis_rows = tuning_df[tuning_df["Axis"] == axis].sort_values(
            ["Combined_Probe_Balanced_Error", "Combined_CV_MSE"]
        )
        selected_alphas[axis] = float(axis_rows.iloc[0]["Alpha"])
    tuning_df["Selected"] = tuning_df.apply(
        lambda row: bool(np.isclose(row["Alpha"], selected_alphas[row["Axis"]])), axis=1
    )

    def assemble(condition, y, run_name):
        x_source = candidate_results[selected_alphas["X"]][condition]
        y_source = candidate_results[selected_alphas["Y"]][condition]
        predictions = np.column_stack([x_source["y_pred"][:, 0], y_source["y_pred"][:, 1]])
        x_folds = x_source["fold_df"]
        y_folds = y_source["fold_df"]
        fold_df = x_folds[[
            "Fold", "LOO_Test_Index", "Train_MSE_X", "Validation_MSE_X", "Learned_Kernel_X",
            "CV_Method", "CV_Folds",
        ]].merge(
            y_folds[["Fold", "LOO_Test_Index", "Train_MSE_Y", "Validation_MSE_Y", "Learned_Kernel_Y"]],
            on=["Fold", "LOO_Test_Index"], how="inner",
        )
        fold_df.insert(0, "Run", run_name)
        fold_df["Shared_Alpha_X"] = selected_alphas["X"]
        fold_df["Shared_Alpha_Y"] = selected_alphas["Y"]
        prediction_df = pd.DataFrame({
            "Run": run_name, "LOO_Test_Index": np.arange(len(y)),
            "Shared_Alpha_X": selected_alphas["X"], "Shared_Alpha_Y": selected_alphas["Y"],
            "True_X": y[:, 0], "True_Y": y[:, 1],
            "Regression_Pred_X": predictions[:, 0], "Regression_Pred_Y": predictions[:, 1],
        })
        metrics = summarize(y, predictions, run_name)
        metrics.update(
            Shared_Alpha_X=selected_alphas["X"], Shared_Alpha_Y=selected_alphas["Y"],
            X_LOO_MSE=float(mean_squared_error(y[:, 0], predictions[:, 0])),
            Y_LOO_MSE=float(mean_squared_error(y[:, 1], predictions[:, 1])),
            CV=(f"{int(args.tuning_cv_folds)}_fold_shared_condition_alpha_tuned_no_grid"
                if getattr(args, "tuning_cv_folds", None) is not None
                else "leave_one_out_shared_condition_alpha_tuned_no_grid"),
            Parameter_Sharing="same_alpha_for_fixation_and_peri_saccade_per_axis",
            PCA_Components=args.pca_components, Grid_Projection=False,
        )
        return {"name": run_name, "y_true": y, "y_pred": predictions, "metrics": metrics,
                "fold_df": fold_df, "prediction_df": prediction_df,
                "alpha_x": selected_alphas["X"], "alpha_y": selected_alphas["Y"]}

    return (
        assemble("fixation", fixY, fix_run_name),
        assemble("peri_saccade", periY, peri_run_name),
        tuning_df,
    )
