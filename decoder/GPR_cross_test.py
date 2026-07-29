import os
import re
import time
import argparse
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.signal as signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from gpr_cross_pca_utils import (
    calculate_aware_unaware_statistics,
    common_condition_units,
    cross_condition_decode_independent_xy_no_grid,
    discover_cross_sessions,
    load_condition_features,
    format_probability,
    rank_available_probes_by_raw_error,
    plot_aware_unaware_signed_error_distributions,
    save_empty_aware_unaware_scatter_plots,
    significance_stars,
    tune_condition_loo_independent_xy_no_grid,
    tune_conditions_shared_independent_xy_no_grid,
)


from thor_ozzy_combined_utils import (
    default_parameter_file,
    discover_combined_sessions,
    find_parameter_row,
    load_parameter_context,
    normalize_animals,
    normalize_directions,
    parse_name_list,
)


# Illustrator-friendly vector output: editable text and uncompressed PDF paths.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["pdf.compression"] = 0


CONDITION_CONFIG = {
    "fixation": {
        "subdir": "fixation",
        "file_suffix": "fix",
        "spike_key": "SpikeProbe_fix",
        "condition_key": "Conditions_fix",
        "label": "Fixation",
    },
    "peri_saccade": {
        "subdir": "peri_saccade",
        "file_suffix": "peri",
        "spike_key": "SpikeProbe_peri",
        "condition_key": "Conditions_peri",
        "label": "Peri-saccade",
    },
}


# -----------------------------
# Helper Functions
# -----------------------------
def find_available_unit_ids(base_data_dir, session_num, condition, max_unit_id=16, excluded_unit_ids=None):
    condition_cfg = CONDITION_CONFIG[condition]
    condition_dir = os.path.join(base_data_dir, session_num, condition_cfg["subdir"])
    pattern = re.compile(
        rf"^spike_probe_{re.escape(str(session_num))}_(\d+)_1_{condition_cfg['file_suffix']}\.mat$"
    )
    excluded_unit_ids = set(excluded_unit_ids or [17])
    unit_ids = []

    if not os.path.isdir(condition_dir):
        raise FileNotFoundError(f"{condition_cfg['label']} directory not found: {condition_dir}")

    for filename in os.listdir(condition_dir):
        match = pattern.match(filename)
        if match:
            unit_id = int(match.group(1))
            if unit_id <= max_unit_id and unit_id not in excluded_unit_ids:
                unit_ids.append(unit_id)

    unit_ids = sorted(set(unit_ids))
    if not unit_ids:
        raise FileNotFoundError(
            f"No {condition_cfg['label']} spike_probe files found in {condition_dir} after filtering "
            f"to unit IDs <= {max_unit_id} and excluding {sorted(excluded_unit_ids)}."
        )
    return unit_ids


def compute_spike_density_function(spike_matrix, kernel_std_samples=10, kernel_size=None):
    if kernel_size is None:
        kernel_size = max(3, int(kernel_std_samples * 6))
    if kernel_size % 2 == 0:
        kernel_size += 1

    gaussian_kernel = signal.windows.gaussian(kernel_size, std=kernel_std_samples, sym=True)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    sdf_matrix = np.apply_along_axis(
        lambda row: signal.convolve(row, gaussian_kernel, mode='same'),
        axis=1,
        arr=spike_matrix.astype(np.float32)
    )
    return sdf_matrix


def bin_sdf_time_window(sdf_matrix, time_window, bin_size_samples=10):
    sdf_window = sdf_matrix[:, time_window].astype(np.float32)
    n_timepoints = sdf_window.shape[1]
    if n_timepoints % bin_size_samples != 0:
        raise ValueError(
            f"Time window length ({n_timepoints}) must be divisible by bin size ({bin_size_samples})."
        )

    n_bins = n_timepoints // bin_size_samples
    return sdf_window.reshape(sdf_window.shape[0], n_bins, bin_size_samples).mean(axis=2)


def load_condition_sdf_data(base_data_dir, session_num, condition, unit_ids, mapping, sdf_time_window, bin_size_samples):
    condition_cfg = CONDITION_CONFIG[condition]
    print(f"Loading {condition_cfg['label']} data with units: {unit_ids}")
    spk_X_list = []
    y_labels = None

    for cid in unit_ids:
        file_path = os.path.join(
            base_data_dir,
            session_num,
            condition_cfg["subdir"],
            f"spike_probe_{session_num}_{cid}_1_{condition_cfg['file_suffix']}.mat",
        )
        mat_data = scipy.io.loadmat(file_path)
        spike_data = np.array(mat_data[condition_cfg["spike_key"]])
        if y_labels is None:
            y_labels = np.squeeze(np.array(mat_data[condition_cfg["condition_key"]]))
        sdf_data = compute_spike_density_function(spike_data)
        binned_sdf_data = bin_sdf_time_window(sdf_data, sdf_time_window, bin_size_samples)
        spk_X_list.append(binned_sdf_data)

    X = np.stack(spk_X_list, axis=1).astype(np.float32)
    y_coords = np.array([mapping[int(val)] for val in y_labels]).astype(np.float32)
    X_reshaped = X.reshape(X.shape[0], -1)
    print(
        f"Finished {condition_cfg['label']} loading. X shape: {X.shape}, "
        f"flattened: {X_reshaped.shape}, Y shape: {y_coords.shape}"
    )
    return X_reshaped, y_coords, y_labels


def build_gpr(args):
    kernel = ConstantKernel(1.0) * RBF(length_scale=args.length_scale)
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=args.alpha,
        normalize_y=True,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state,
    )


def safe_pearson_avg(y_true, y_pred):
    pearson_values = []
    for dim_idx in range(y_true.shape[1]):
        if len(y_true) < 2 or np.std(y_true[:, dim_idx]) == 0 or np.std(y_pred[:, dim_idx]) == 0:
            pearson_values.append(np.nan)
        else:
            pearson_values.append(pearsonr(y_true[:, dim_idx], y_pred[:, dim_idx])[0])
    return float(np.nanmean(pearson_values))


def summarize_predictions(y_true, y_pred, name):
    return {
        "Run": name,
        "Num_Test_Samples": len(y_true),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Pearson_avg": safe_pearson_avg(y_true, y_pred),
        "Mean_Euclidean_Error": float(np.mean(np.linalg.norm(y_pred - y_true, axis=1))),
    }


def get_avg_pred_locs(y_true, y_pred):
    unique_locs = np.unique(y_true, axis=0)
    trial_counts = np.array([
        np.sum(np.all(y_true == loc, axis=1))
        for loc in unique_locs
    ])
    avg_pred_locs = np.array([
        np.mean(y_pred[np.all(y_true == loc, axis=1)], axis=0)
        for loc in unique_locs
    ])
    return unique_locs, avg_pred_locs, trial_counts


def avg_pred_locs_to_df(unique_locs, avg_pred_locs, trial_counts, run_name):
    rows = []
    for loc, avg_pred, trial_count in zip(unique_locs, avg_pred_locs, trial_counts):
        rows.append({
            "Run": run_name,
            "True_X": loc[0],
            "True_Y": loc[1],
            "Num_Trials": int(trial_count),
            "Avg_Pred_X": avg_pred[0],
            "Avg_Pred_Y": avg_pred[1],
            "Distance_True_to_Avg_Pred": float(np.linalg.norm(avg_pred - loc)),
        })
    return pd.DataFrame(rows)


def generate_visualizations(y_true, y_pred, viz_dir, prefix):
    print(f"  - Generating visualizations for '{prefix}'...")
    os.makedirs(viz_dir, exist_ok=True)
    viz_axis_limits, viz_axis_ticks = (-0.5, 2.5), [0, 1, 2]

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true[:, 0], y_true[:, 1], color='blue', label="True Location", alpha=0.5)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label="Predicted Location", alpha=0.5)
    for idx in range(len(y_true)):
        plt.plot([y_true[idx, 0], y_pred[idx, 0]], [y_true[idx, 1], y_pred[idx, 1]], 'k--', alpha=0.3)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(viz_axis_limits)
    plt.ylim(viz_axis_limits)
    plt.xticks(viz_axis_ticks)
    plt.yticks(viz_axis_ticks)
    plt.legend()
    plt.grid(True)
    plt.title(f"Scatter Plot ({prefix})")
    plt.savefig(os.path.join(viz_dir, f"{prefix}_scatter.png"))
    plt.close()

    unique_locs, avg_pred_locs, _ = get_avg_pred_locs(y_true, y_pred)
    loc_error_grid = np.full((3, 3), np.nan)
    for probe_loc in unique_locs:
        mask = np.all(y_true == probe_loc, axis=1)
        loc_error_grid[int(probe_loc[0]), int(probe_loc[1])] = np.mean(
            np.linalg.norm(y_pred[mask] - probe_loc, axis=1)
        )

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(loc_error_grid.T, annot=True, fmt=".2f", cmap="coolwarm",
                     cbar_kws={'label': 'Mean Euclidean Error'}, vmin=0,
                     xticklabels=viz_axis_ticks, yticklabels=viz_axis_ticks)
    ax.invert_yaxis()
    plt.title(f"Mean Euclidean Dist. Error ({prefix})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(os.path.join(viz_dir, f"{prefix}_error_heatmap.png"))
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.scatter(unique_locs[:, 0], unique_locs[:, 1], color='blue', label="True Probe Location", alpha=0.5, s=100)
    plt.scatter(avg_pred_locs[:, 0], avg_pred_locs[:, 1], color='green',
                label="Avg Prediction Location", alpha=0.7, marker='s', s=100)
    for loc_idx in range(len(unique_locs)):
        plt.plot([unique_locs[loc_idx, 0], avg_pred_locs[loc_idx, 0]],
                 [unique_locs[loc_idx, 1], avg_pred_locs[loc_idx, 1]], 'k--', alpha=0.3)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(viz_axis_limits)
    plt.ylim(viz_axis_limits)
    plt.xticks(viz_axis_ticks)
    plt.yticks(viz_axis_ticks)
    plt.legend()
    plt.grid(True)
    plt.title(f"Avg Pred Loc per Probe ({prefix})")
    plt.savefig(os.path.join(viz_dir, f"{prefix}_avg_loc_per_probe.png"))
    plt.close()


def plot_avg_prediction_shift_arrows(start_locs, start_avg, end_locs, end_avg, viz_dir, prefix, title, start_label, end_label):
    os.makedirs(viz_dir, exist_ok=True)
    start_lookup = {tuple(loc): avg for loc, avg in zip(start_locs, start_avg)}
    end_lookup = {tuple(loc): avg for loc, avg in zip(end_locs, end_avg)}
    common_locs = sorted(set(start_lookup) & set(end_lookup))
    if not common_locs:
        print(f"  - Skipping '{prefix}' arrow plot: no common probe locations.")
        return pd.DataFrame()

    starts = np.array([start_lookup[loc] for loc in common_locs])
    ends = np.array([end_lookup[loc] for loc in common_locs])
    true_locs = np.array(common_locs, dtype=np.float32)
    deltas = ends - starts
    shift_lengths = np.linalg.norm(deltas, axis=1)

    component_configs = [
        (0, "DX", "tab:red", "X shift component"),
        (1, "DY", "tab:green", "Y shift component"),
    ]
    for axis, suffix, color, component_label in component_configs:
        component_ends = starts.copy()
        component_ends[:, axis] = ends[:, axis]
        plt.figure(figsize=(5.5, 5.0))
        plt.scatter(true_locs[:, 0], true_locs[:, 1], color="black", marker="x", s=120,
                    label="True Probe Location")
        plt.scatter(starts[:, 0], starts[:, 1], color="tab:blue", marker="o", s=85, label=start_label)
        plt.scatter(component_ends[:, 0], component_ends[:, 1], color=color, marker="s", s=85,
                    label=f"{suffix} endpoint")
        for loc, start, component_end, delta in zip(true_locs, starts, component_ends, deltas[:, axis]):
            plt.annotate("", xy=component_end, xytext=start,
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.85))
            midpoint = (start + component_end) / 2
            plt.text(midpoint[0] + 0.03, midpoint[1] + 0.03, f"{suffix.lower()}={delta:+.2f}",
                     fontsize=7, color=color)
            plt.plot([loc[0], start[0]], [loc[1], start[1]], linestyle=":",
                     color="tab:blue", alpha=0.35)
        plt.plot([], [], color=color, lw=2, label=component_label)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xlim((-0.5, 2.5))
        plt.ylim((-0.5, 2.5))
        plt.xticks([0, 1, 2])
        plt.yticks([0, 1, 2])
        plt.legend()
        plt.grid(True)
        plt.title(f"{title} ({suffix})")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{prefix}_avg_prediction_shift_{suffix.lower()}_arrows.png"))
        plt.close()

    return pd.DataFrame({
        "Comparison": prefix,
        "True_X": true_locs[:, 0],
        "True_Y": true_locs[:, 1],
        "Start_Avg_Pred_X": starts[:, 0],
        "Start_Avg_Pred_Y": starts[:, 1],
        "End_Avg_Pred_X": ends[:, 0],
        "End_Avg_Pred_Y": ends[:, 1],
        "Delta_X": deltas[:, 0],
        "Delta_Y": deltas[:, 1],
        "Shift_Length": shift_lengths,
    })


def run_cross_condition_decoding(trainX, trainY, testX, testY, args, run_name):
    print(f"Starting {run_name}. Train samples: {len(trainY)}, test samples: {len(testY)}")
    gpr_regressor = build_gpr(args)
    gpr_regressor.fit(trainX, trainY)
    y_pred_train = gpr_regressor.predict(trainX)
    y_pred_test = gpr_regressor.predict(testX)

    metrics = summarize_predictions(testY, y_pred_test, run_name)
    metrics["Train_MSE"] = mean_squared_error(trainY, y_pred_train)
    metrics["CV"] = "train_full_condition_test_other_condition"
    print(f"{run_name} complete. MSE: {metrics['MSE']:.4f}")

    prediction_df = pd.DataFrame({
        "Run": run_name,
        "Test_Index": np.arange(len(testY)),
        "True_X": testY[:, 0],
        "True_Y": testY[:, 1],
        "Regression_Pred_X": y_pred_test[:, 0],
        "Regression_Pred_Y": y_pred_test[:, 1],
    })
    return {
        "name": run_name,
        "y_true": testY,
        "y_pred": y_pred_test,
        "metrics": metrics,
        "prediction_df": prediction_df,
    }


def plot_cross_session_average_2d(session_df, average_df, title, output_path, average_label):
    plt.figure(figsize=(5, 5))
    for _, rows in session_df.groupby("Session"):
        plt.scatter(rows["Avg_Pred_X"], rows["Avg_Pred_Y"], color="gray", alpha=0.18, s=25)
    true_locs = average_df[["True_X", "True_Y"]].to_numpy()
    pred_locs = average_df[["Avg_Pred_X", "Avg_Pred_Y"]].to_numpy()
    plt.scatter(true_locs[:, 0], true_locs[:, 1], color="blue", label="True Location", s=110, alpha=0.7)
    plt.scatter(pred_locs[:, 0], pred_locs[:, 1], color="green", label=average_label, marker="s", s=110)
    for true_loc, pred_loc in zip(true_locs, pred_locs):
        plt.plot([true_loc[0], pred_loc[0]], [true_loc[1], pred_loc[1]], "k--", alpha=0.45)
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cross_session_shift_2d(session_shift_df, _average_shift_df, title, output_path):
    true_locations = session_shift_df[["True_X", "True_Y"]].drop_duplicates()
    output_stem, output_ext = os.path.splitext(output_path)
    component_configs = [
        ("Delta_X", "DX", "tab:red", "Session X shift"),
        ("Delta_Y", "DY", "tab:green", "Session Y shift"),
    ]
    for delta_column, suffix, color, label in component_configs:
        plt.figure(figsize=(6, 6))
        for _, row in session_shift_df.iterrows():
            start = (row["True_X"], row["True_Y"])
            if delta_column == "Delta_X":
                end = (row["True_X"] + row[delta_column], row["True_Y"])
            else:
                end = (row["True_X"], row["True_Y"] + row[delta_column])
            plt.annotate("", xy=end, xytext=start,
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.25))
            plt.scatter(end[0], end[1], color=color, s=18, alpha=0.35, zorder=2)
        plt.scatter(true_locations["True_X"], true_locations["True_Y"],
                    color="blue", s=90, label="True location", zorder=3)
        plt.plot([], [], color=color, lw=2, alpha=0.7, label=label)
        plt.scatter([], [], color=color, s=25, alpha=0.6, label=f"Session {suffix} endpoint")
        plt.title(f"{title} ({suffix})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 2.5)
        plt.xticks([0, 1, 2])
        plt.yticks([0, 1, 2])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_stem}_{suffix.lower()}{output_ext}")
        plt.close()


def plot_average_vector_magnitude_2d(average_shift_df, title, output_path):
    plt.figure(figsize=(6, 6))
    for _, row in average_shift_df.iterrows():
        plt.arrow(
            row["True_X"], row["True_Y"],
            row["Magnitude_Plot_Delta_X"], row["Magnitude_Plot_Delta_Y"],
            color="purple", alpha=0.9, width=0.012, head_width=0.08,
            length_includes_head=True,
        )
        end_x = row["True_X"] + row["Magnitude_Plot_Delta_X"]
        end_y = row["True_Y"] + row["Magnitude_Plot_Delta_Y"]
        plt.scatter(end_x, end_y, color="purple", marker="s", s=45, zorder=3)
        plt.text(end_x + 0.025, end_y + 0.025, f"{row['Mean_Vector_Magnitude']:.2f}",
                 color="purple", fontsize=8)
    plt.scatter(average_shift_df["True_X"], average_shift_df["True_Y"],
                color="blue", s=90, label="True location", zorder=3)
    plt.plot([], [], color="purple", lw=3, label="Mean vector magnitude")
    plt.scatter([], [], color="purple", marker="s", s=45, label="Mean vector endpoint")
    plt.title(f"{title} (Mean Vector Magnitude)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_average_component_magnitude_2d(average_shift_df, title, output_stem):
    component_configs = [
        ("X", "Plot_Delta_X", "Mean_Abs_Delta_X", "tab:red"),
        ("Y", "Plot_Delta_Y", "Mean_Abs_Delta_Y", "tab:green"),
    ]
    for axis, plot_delta_column, magnitude_column, color in component_configs:
        plt.figure(figsize=(6, 6))
        for _, row in average_shift_df.iterrows():
            start_x, start_y = row["True_X"], row["True_Y"]
            if axis == "X":
                delta_x, delta_y = row[plot_delta_column], 0.0
            else:
                delta_x, delta_y = 0.0, row[plot_delta_column]
            plt.arrow(start_x, start_y, delta_x, delta_y, color=color, alpha=0.9,
                      width=0.012, head_width=0.08, length_includes_head=True)
            end_x, end_y = start_x + delta_x, start_y + delta_y
            plt.scatter(end_x, end_y, color=color, marker="s", s=45, zorder=3)
            plt.text(end_x + 0.025, end_y + 0.025, f"{row[magnitude_column]:.2f}",
                     color=color, fontsize=8)
        plt.scatter(average_shift_df["True_X"], average_shift_df["True_Y"],
                    color="blue", s=90, label="True location", zorder=3)
        plt.plot([], [], color=color, lw=3, label=f"Mean |D{axis}| magnitude")
        plt.scatter([], [], color=color, marker="s", s=45, label=f"Mean D{axis} endpoint")
        plt.title(f"{title} (Mean |D{axis}| Magnitude)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 2.5)
        plt.xticks([0, 1, 2])
        plt.yticks([0, 1, 2])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_stem}_mean_d{axis.lower()}_magnitude.png")
        plt.close()


def create_shift_average_outputs(metrics_df, session_avg_df, root_result_dir):
    comparisons = [
        ("fix_train_fix_test_to_peri_test", "fixation_train_fixation_test_LOO",
         "fixation_train_peri_saccade_test"),
        ("fix_train_test_to_peri_train_test", "fixation_train_fixation_test_LOO",
         "peri_saccade_train_peri_saccade_test_LOO"),
    ]
    output_tables = []
    for comparison, start_run, end_run in comparisons:
        start_df = session_avg_df[session_avg_df["Run"] == start_run].rename(columns={
            "Avg_Pred_X": "Start_Avg_Pred_X", "Avg_Pred_Y": "Start_Avg_Pred_Y",
        })
        end_df = session_avg_df[session_avg_df["Run"] == end_run].rename(columns={
            "Avg_Pred_X": "End_Avg_Pred_X", "Avg_Pred_Y": "End_Avg_Pred_Y",
        })
        shift_keys = ["Animal", "Session", "Raw_Session", "Saccade_Direction", "True_X", "True_Y"]
        shifts = start_df[shift_keys + ["Start_Avg_Pred_X", "Start_Avg_Pred_Y"]].merge(
            end_df[shift_keys + ["End_Avg_Pred_X", "End_Avg_Pred_Y"]],
            on=shift_keys, how="inner",
        )
        direction_sign = np.where(shifts["Saccade_Direction"].eq("left"), -1.0, 1.0)
        shifts["Delta_X"] = direction_sign * (
            shifts["End_Avg_Pred_X"] - shifts["Start_Avg_Pred_X"]
        )
        shifts["Delta_Y"] = shifts["End_Avg_Pred_Y"] - shifts["Start_Avg_Pred_Y"]
        shifts["Abs_Delta_X"] = np.abs(shifts["Delta_X"])
        shifts["Abs_Delta_Y"] = np.abs(shifts["Delta_Y"])
        shifts["Vector_Magnitude"] = np.hypot(shifts["Delta_X"], shifts["Delta_Y"])
        safe_magnitude = shifts["Vector_Magnitude"].replace(0, np.nan)
        shifts["Unit_Direction_X"] = (shifts["Delta_X"] / safe_magnitude).fillna(0.0)
        shifts["Unit_Direction_Y"] = (shifts["Delta_Y"] / safe_magnitude).fillna(0.0)
        ranked = metrics_df[metrics_df["Run"] == end_run].sort_values(["Mean_Euclidean_Error", "MSE"])
        num_top = int(np.ceil(len(ranked) * 0.5))
        top_ids = set(ranked.head(num_top)["Session"].astype(str))
        scopes = [
            ("All_Sessions", shifts, f"All-Session Shift Vectors\n{comparison}"),
            ("Top_50_Percent", shifts[shifts["Session"].astype(str).isin(top_ids)],
             f"Top 50% Session Shift Vectors (n={num_top})\n{comparison}"),
        ]
        for scope, selected, title in scopes:
            average = selected.groupby(["True_X", "True_Y"], as_index=False).agg(
                Mean_Delta_X=("Delta_X", "mean"),
                Mean_Delta_Y=("Delta_Y", "mean"),
                Mean_Abs_Delta_X=("Abs_Delta_X", "mean"),
                Mean_Abs_Delta_Y=("Abs_Delta_Y", "mean"),
                Abs_Delta_X_SD=("Abs_Delta_X", "std"),
                Abs_Delta_Y_SD=("Abs_Delta_Y", "std"),
                Mean_Unit_Direction_X=("Unit_Direction_X", "mean"),
                Mean_Unit_Direction_Y=("Unit_Direction_Y", "mean"),
                Mean_Vector_Magnitude=("Vector_Magnitude", "mean"),
                Vector_Magnitude_SD=("Vector_Magnitude", "std"),
                Num_Sessions=("Session", "nunique"),
            )
            direction_consistency = np.hypot(
                average["Mean_Unit_Direction_X"], average["Mean_Unit_Direction_Y"]
            )
            mean_resultant_magnitude = np.hypot(average["Mean_Delta_X"], average["Mean_Delta_Y"])
            average["Plot_Delta_X"] = np.sign(average["Mean_Delta_X"]) * average["Mean_Abs_Delta_X"]
            average["Plot_Delta_Y"] = np.sign(average["Mean_Delta_Y"]) * average["Mean_Abs_Delta_Y"]
            safe_resultant = mean_resultant_magnitude.replace(0, np.nan)
            average["Magnitude_Plot_Delta_X"] = (
                average["Mean_Delta_X"] / safe_resultant * average["Mean_Vector_Magnitude"]
            ).fillna(0.0)
            average["Magnitude_Plot_Delta_Y"] = (
                average["Mean_Delta_Y"] / safe_resultant * average["Mean_Vector_Magnitude"]
            ).fillna(0.0)
            average["Mean_Resultant_Vector_Magnitude"] = mean_resultant_magnitude
            average["Direction_Consistency"] = direction_consistency
            average.insert(0, "Comparison", comparison)
            average.insert(1, "Scope", scope)
            output_tables.append(average)
            suffix = "all_sessions" if scope == "All_Sessions" else "top_50_percent_sessions"
            plot_cross_session_shift_2d(
                selected, average, title,
                os.path.join(root_result_dir, f"{comparison}_{suffix}_average_shift_arrows.png"),
            )
            plot_average_vector_magnitude_2d(
                average, title,
                os.path.join(root_result_dir, f"{comparison}_{suffix}_vector_magnitude_average.png"),
            )
            plot_average_component_magnitude_2d(
                average, title,
                os.path.join(root_result_dir, f"{comparison}_{suffix}"),
            )
    return pd.concat(output_tables, ignore_index=True)


def create_marginal_scatter_axes():
    fig = plt.figure(figsize=(7.5, 7.5), layout="constrained")
    grid = fig.add_gridspec(4, 4, hspace=0.06, wspace=0.06)
    scatter_ax = fig.add_subplot(grid[1:, :3])
    hist_x_ax = fig.add_subplot(grid[0, :3], sharex=scatter_ax)
    hist_y_ax = fig.add_subplot(grid[1:, 3], sharey=scatter_ax)
    hist_x_ax.tick_params(axis="x", labelbottom=False)
    hist_y_ax.tick_params(axis="y", labelleft=False)
    return fig, scatter_ax, hist_x_ax, hist_y_ax


def add_illustrator_editable_points(
        ax, x_values, y_values, color, label, marker_radius):
    """Draw each data point as an independent opaque vector path."""
    for point_index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        point = Circle(
            (float(x_value), float(y_value)), marker_radius,
            facecolor=color, edgecolor="none", linewidth=0, alpha=1.0,
            clip_on=True,
        )
        point.set_gid(f"{label}_data_point_{point_index:04d}")
        ax.add_patch(point)
    ax.plot(
        [], [], linestyle="none", marker="o", markersize=np.sqrt(42),
        markerfacecolor=color, markeredgecolor="none", label=label,
    )


def add_marginal_histograms(hist_x_ax, hist_y_ax, x_values, y_values, plot_limit):
    bins = np.linspace(-plot_limit, plot_limit, 16)
    hist_x_ax.hist(x_values, bins=bins, color="tab:blue", alpha=0.7, edgecolor="white")
    hist_y_ax.hist(y_values, bins=bins, orientation="horizontal",
                   color="tab:orange", alpha=0.7, edgecolor="white")
    hist_x_ax.axvline(0, color="black", linewidth=0.8, alpha=0.45)
    hist_y_ax.axhline(0, color="black", linewidth=0.8, alpha=0.45)
    hist_x_ax.set_ylabel("Count")
    hist_y_ax.set_xlabel("Count")
    hist_x_ax.grid(True, alpha=0.2)
    hist_y_ax.grid(True, alpha=0.2)


def summarize_top50_aware_unaware(scopes, session_key_columns, statistics_df):
    """Create a compact Top 50% signed-delta summary at the session level."""
    top50_scope = "top_50_percent_raw_probes_min_3"
    top50_matches = [
        selected_df for scope_suffix, selected_df, _ in scopes
        if scope_suffix == top50_scope
    ]
    if len(top50_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {top50_scope!r} scope, found {len(top50_matches)}."
        )
    top50_df = top50_matches[0]
    rows = []

    for monkey_group in ("All monkeys", "Thor", "Ozzy"):
        group_df = (
            top50_df
            if monkey_group == "All monkeys"
            else top50_df[top50_df["Animal"].eq(monkey_group)]
        )
        for axis in ("X", "Y"):
            unaware_column = f"Unaware_Delta_{axis}"
            aware_column = f"Aware_Delta_{axis}"
            finite_rows = group_df[
                np.isfinite(group_df[unaware_column])
                & np.isfinite(group_df[aware_column])
            ]
            grouped = finite_rows.groupby(session_key_columns, as_index=False)[
                [unaware_column, aware_column]
            ].mean()
            unaware = grouped[unaware_column].to_numpy(dtype=float)
            aware = grouped[aware_column].to_numpy(dtype=float)

            def quartile_summary(values):
                if len(values) == 0:
                    return np.nan, np.nan, np.nan, np.nan, ""
                q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
                return (
                    float(median), float(q1), float(q3), float(q3 - q1),
                    f"{median:.6g} [{q1:.6g}, {q3:.6g}]",
                )

            unaware_median, unaware_q1, unaware_q3, unaware_iqr, unaware_text = (
                quartile_summary(unaware)
            )
            aware_median, aware_q1, aware_q3, aware_iqr, aware_text = (
                quartile_summary(aware)
            )
            matching_statistics = statistics_df[
                statistics_df["Monkey_Group"].eq(monkey_group)
                & statistics_df["Scope"].eq(top50_scope)
                & statistics_df["Axis"].eq(axis)
                & statistics_df["Measure"].eq("Signed_Delta")
            ]
            if len(matching_statistics) != 1:
                raise RuntimeError(
                    "Expected one signed-delta statistics row for "
                    f"{monkey_group}, axis {axis}, {top50_scope}; "
                    f"found {len(matching_statistics)}."
                )
            stat = matching_statistics.iloc[0]
            rows.append({
                "Scope": top50_scope,
                "Monkey_Group": monkey_group,
                "Axis": axis,
                "Independent_Unit": " + ".join(session_key_columns),
                "N_Independent_Units": len(grouped),
                "N_Session_Probe_Pairs": len(finite_rows),
                "Unaware_Median": unaware_median,
                "Unaware_Q1": unaware_q1,
                "Unaware_Q3": unaware_q3,
                "Unaware_IQR_Width": unaware_iqr,
                "Unaware_Median_[Q1,Q3]": unaware_text,
                "Unaware_vs_Zero_Wilcoxon_P": stat["Unaware_Bias_Wilcoxon_P"],
                "Aware_Median": aware_median,
                "Aware_Q1": aware_q1,
                "Aware_Q3": aware_q3,
                "Aware_IQR_Width": aware_iqr,
                "Aware_Median_[Q1,Q3]": aware_text,
                "Aware_vs_Zero_Wilcoxon_P": stat["Aware_Bias_Wilcoxon_P"],
                "Aware_vs_Unaware_Wilcoxon_Statistic": stat["Wilcoxon_Statistic"],
                "Aware_vs_Unaware_Wilcoxon_P": stat["Wilcoxon_P"],
            })
    return pd.DataFrame(rows)


def create_aware_vs_unaware_signed_scatter(
        metrics_df, session_avg_df, trial_predictions_df, root_result_dir):
    os.makedirs(root_result_dir, exist_ok=True)
    baseline_run = "fixation_train_fixation_test_LOO"
    unaware_run = "fixation_train_peri_saccade_test"
    aware_run = "peri_saccade_train_peri_saccade_test_LOO"
    keys = ["Animal", "Session", "Saccade_Direction", "True_X", "True_Y"]

    baseline = session_avg_df[session_avg_df["Run"] == baseline_run][
        keys + ["Num_Trials", "Avg_Pred_X", "Avg_Pred_Y"]
    ].rename(columns={"Num_Trials": "Baseline_Num_Trials",
                      "Avg_Pred_X": "Baseline_Pred_X", "Avg_Pred_Y": "Baseline_Pred_Y"})
    unaware = session_avg_df[session_avg_df["Run"] == unaware_run][
        keys + ["Num_Trials", "Avg_Pred_X", "Avg_Pred_Y"]
    ].rename(columns={"Num_Trials": "Unaware_Num_Trials",
                      "Avg_Pred_X": "Unaware_Pred_X", "Avg_Pred_Y": "Unaware_Pred_Y"})
    aware = session_avg_df[session_avg_df["Run"] == aware_run][
        keys + ["Num_Trials", "Avg_Pred_X", "Avg_Pred_Y"]
    ].rename(columns={"Num_Trials": "Aware_Num_Trials",
                      "Avg_Pred_X": "Aware_Pred_X", "Avg_Pred_Y": "Aware_Pred_Y"})
    scatter_df = baseline.merge(unaware, on=keys, how="inner").merge(aware, on=keys, how="inner")
    scatter_df = scatter_df[
        (
            (
                scatter_df["Saccade_Direction"].eq("left")
                & scatter_df["True_X"].isin([0, 1])
            )
            | (
                scatter_df["Saccade_Direction"].eq("right")
                & scatter_df["True_X"].isin([1, 2])
            )
        )
        & (scatter_df["Baseline_Num_Trials"] > 2)
        & (scatter_df["Unaware_Num_Trials"] > 2)
        & (scatter_df["Aware_Num_Trials"] > 2)
    ].copy()
    for axis in ("X", "Y"):
        axis_sign = (
            np.where(scatter_df["Saccade_Direction"].eq("left"), -1.0, 1.0)
            if axis == "X" else 1.0
        )
        scatter_df[f"Unaware_Delta_{axis}"] = (
            axis_sign * (scatter_df[f"Unaware_Pred_{axis}"] - scatter_df[f"Baseline_Pred_{axis}"])
        )
        scatter_df[f"Aware_Delta_{axis}"] = (
            axis_sign * (scatter_df[f"Aware_Pred_{axis}"] - scatter_df[f"Baseline_Pred_{axis}"])
        )
        scatter_df[f"Unaware_Abs_Delta_{axis}"] = np.abs(scatter_df[f"Unaware_Delta_{axis}"])
        scatter_df[f"Aware_Abs_Delta_{axis}"] = np.abs(scatter_df[f"Aware_Delta_{axis}"])

    raw_ranking = rank_available_probes_by_raw_error(
        trial_predictions_df, baseline_run, keys, scatter_df[keys]
    )
    raw_ranking["Animal_Raw_Probe_Rank"] = (
        raw_ranking.groupby("Animal", sort=False).cumcount() + 1
    )
    ranking_columns = [
        "Fixation_Probe_Avg_Euclidean_Error", "Fixation_Probe_Num_Trials",
        "Raw_Probe_Rank", "Animal_Raw_Probe_Rank",
    ]
    scatter_df = scatter_df.merge(
        raw_ranking[keys + ranking_columns],
        on=keys, how="inner", validate="many_to_one",
    )
    session_key_columns = [column for column in keys if column not in ("True_X", "True_Y")]
    minimum_top_probes_per_session = 3
    top_scope_specs = []
    for top_percent in (60, 50, 40, 30, 20, 10):
        candidate_limits = {
            animal: max(1, int(np.ceil(len(animal_rows) * top_percent / 100.0)))
            for animal, animal_rows in raw_ranking.groupby("Animal", sort=False)
        }
        top_candidates = raw_ranking[
            raw_ranking["Animal_Raw_Probe_Rank"]
            <= raw_ranking["Animal"].map(candidate_limits)
        ].copy()
        num_top_candidates = len(top_candidates)
        candidate_counts = top_candidates.groupby("Animal").size()
        candidate_count_text = ", ".join(
            f"{animal}={int(candidate_counts.get(animal, 0))}"
            for animal in ("Thor", "Ozzy")
            if animal in candidate_limits
        )
        qualifying_counts = top_candidates.groupby(session_key_columns).size()
        eligible_sessions = qualifying_counts[
            qualifying_counts >= minimum_top_probes_per_session
        ]
        if len(session_key_columns) == 1:
            filtered_probes = top_candidates[
                top_candidates[session_key_columns[0]].isin(eligible_sessions.index)
            ]
        else:
            eligible_index = pd.MultiIndex.from_tuples(
                list(eligible_sessions.index), names=session_key_columns
            )
            filtered_probes = top_candidates[
                pd.MultiIndex.from_frame(top_candidates[session_key_columns]).isin(eligible_index)
            ]
        flag_column = f"Is_Top_{top_percent}_Raw_Probe_Eligible_Session"
        filtered_keys = pd.MultiIndex.from_frame(filtered_probes[keys])
        scatter_df[flag_column] = pd.MultiIndex.from_frame(scatter_df[keys]).isin(filtered_keys)
        top_scope_specs.append((
            top_percent, num_top_candidates, candidate_count_text,
            len(filtered_probes), len(eligible_sessions), flag_column
        ))

    scopes = [("", scatter_df, "All Sessions")]
    scopes.extend(
        (
            f"top_{top_percent}_percent_raw_probes_min_3",
            scatter_df[scatter_df[flag_column]],
            f"Top {top_percent}% Raw Probe Error per Monkey + Min 3/Session "
            f"(candidates={num_top_candidates}; {candidate_count_text}, "
            f"retained={num_retained_probes}, sessions={num_eligible_sessions})",
        )
        for top_percent, num_top_candidates, candidate_count_text,
        num_retained_probes, num_eligible_sessions, flag_column in top_scope_specs
    )
    all_monkeys_statistics_df = calculate_aware_unaware_statistics(
        scopes, session_key_columns
    )
    all_monkeys_statistics_df.insert(0, "Monkey_Group", "All monkeys")
    grouped_statistics = [all_monkeys_statistics_df]
    for animal in ("Thor", "Ozzy"):
        animal_scopes = [
            (scope_suffix, selected_df[selected_df["Animal"].eq(animal)], scope_title)
            for scope_suffix, selected_df, scope_title in scopes
        ]
        animal_statistics_df = calculate_aware_unaware_statistics(
            animal_scopes, session_key_columns
        )
        animal_statistics_df.insert(0, "Monkey_Group", animal)
        grouped_statistics.append(animal_statistics_df)
    statistics_df = pd.concat(grouped_statistics, ignore_index=True)
    for scope_suffix, selected_df, scope_title in scopes:
        if selected_df.empty:
            empty_scope_title = f"{scope_title}; direction-relative 6 probes"
            save_empty_aware_unaware_scatter_plots(
                empty_scope_title, scope_suffix, root_result_dir, "session-probe means",
                file_extensions=("pdf", "svg"),
            )
            plot_aware_unaware_signed_error_distributions(
                selected_df, empty_scope_title, scope_suffix, root_result_dir,
                point_level="session-probe means", axes=("X", "Y"),
                statistics_df=all_monkeys_statistics_df,
            )
            continue
        for axis in ("X", "Y"):
            x_column = f"Unaware_Delta_{axis}"
            y_column = f"Aware_Delta_{axis}"
            plot_df = selected_df[
                np.isfinite(selected_df[x_column]) & np.isfinite(selected_df[y_column])
            ].copy()
            if plot_df.empty:
                print(f"Skipping {scope_title} Delta {axis}: no finite paired predictions.")
                continue
            fig, scatter_ax, hist_x_ax, hist_y_ax = create_marginal_scatter_axes()
            plt.sca(scatter_ax)
            max_abs_value = float(max(
                plot_df[x_column].abs().max(), plot_df[y_column].abs().max(), 1e-6
            ))
            plot_limit = max_abs_value * 1.10
            marker_radius = plot_limit * 0.02
            animal_colors = {"Ozzy": "tab:blue", "Thor": "tab:orange"}
            for animal, color in animal_colors.items():
                animal_df = plot_df[plot_df["Animal"].eq(animal)]
                if animal_df.empty:
                    continue
                add_illustrator_editable_points(
                    scatter_ax,
                    animal_df[x_column].to_numpy(),
                    animal_df[y_column].to_numpy(),
                    color, animal, marker_radius,
                )
            plt.plot([-plot_limit, plot_limit], [-plot_limit, plot_limit],
                     "k--", alpha=0.55, label="Aware = Unaware")
            plt.axhline(0, color="black", linewidth=0.8, alpha=0.45)
            plt.axvline(0, color="black", linewidth=0.8, alpha=0.45)
            if (len(plot_df) > 1 and plot_df[x_column].std() > 0
                    and plot_df[y_column].std() > 0):
                correlation = float(pearsonr(plot_df[x_column], plot_df[y_column])[0])
            else:
                correlation = np.nan
            stats_row = all_monkeys_statistics_df[
                (all_monkeys_statistics_df["Scope"] == (scope_suffix or "all_probes"))
                & (all_monkeys_statistics_df["Axis"] == axis)
                & (all_monkeys_statistics_df["Measure"] == "Signed_Delta")
            ].iloc[0]
            plt.text(
                0.04, 0.96,
                f"n={len(plot_df)} points, {int(stats_row['N_Independent_Units'])} sessions\n"
                f"r={correlation:.3f}\n"
                f"Unaware vs 0: {significance_stars(stats_row['Unaware_Bias_Wilcoxon_P'])} "
                f"(p={format_probability(stats_row['Unaware_Bias_Wilcoxon_P'])})\n"
                f"Aware vs 0: {significance_stars(stats_row['Aware_Bias_Wilcoxon_P'])} "
                f"(p={format_probability(stats_row['Aware_Bias_Wilcoxon_P'])})\n"
                f"Aware vs Unaware: {significance_stars(stats_row['Wilcoxon_P'])} "
                f"(p={format_probability(stats_row['Wilcoxon_P'])})",
                transform=plt.gca().transAxes, va="top",
            )
            direction_text = ("Opposite saccade direction (-) / Saccade direction (+)"
                              if axis == "X" else "Down (-) / Up (+)")
            plt.xlabel(f"Unaware Signed Delta {axis}  [{direction_text}]")
            plt.ylabel(f"Aware Signed Delta {axis}  [{direction_text}]")
            plt.xlim(-plot_limit, plot_limit)
            plt.ylim(-plot_limit, plot_limit)
            plt.gca().set_aspect("equal", adjustable="box")
            add_marginal_histograms(
                hist_x_ax, hist_y_ax, plot_df[x_column], plot_df[y_column], plot_limit
            )
            fig.suptitle(
                f"Aware vs Unaware Signed Mislocalization (Delta {axis})\n"
                f"{scope_title}; direction-relative 6 probes; Session x Probe Points"
            )
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, ncol=2, loc="lower right")
            output_stem = os.path.join(
                root_result_dir,
                f"aware_vs_unaware_delta_{axis.lower()}_signed_scatter"
                f"{'_' + scope_suffix if scope_suffix else ''}",
            )
            for file_extension in ("pdf", "svg"):
                fig.savefig(f"{output_stem}.{file_extension}")
            plt.close(fig)
        plot_aware_unaware_signed_error_distributions(
            selected_df, f"{scope_title}; direction-relative 6 probes", scope_suffix, root_result_dir,
            point_level="session-probe means", axes=("X", "Y"),
            statistics_df=all_monkeys_statistics_df,
        )
    top50_summary_df = summarize_top50_aware_unaware(
        scopes, session_key_columns, statistics_df
    )
    return scatter_df, statistics_df, top50_summary_df


def create_all_session_average_outputs(metrics_df, session_avg_df, root_result_dir):
    average_tables, top_session_tables = [], []
    for run_name, run_avg_df in session_avg_df.groupby("Run"):
        run_metrics = metrics_df[metrics_df["Run"] == run_name].copy()
        run_metrics = run_metrics.sort_values(["Mean_Euclidean_Error", "MSE"])
        num_top = int(np.ceil(len(run_metrics) * 0.5))
        top_metrics = run_metrics.head(num_top).copy()
        top_ids = set(top_metrics["Session"].astype(str))
        top_avg_df = run_avg_df[run_avg_df["Session"].astype(str).isin(top_ids)].copy()

        all_average = run_avg_df.groupby(["True_X", "True_Y"], as_index=False).agg(
            Avg_Pred_X=("Avg_Pred_X", "mean"), Avg_Pred_Y=("Avg_Pred_Y", "mean"),
            Num_Sessions=("Session", "nunique"),
        )
        top_average = top_avg_df.groupby(["True_X", "True_Y"], as_index=False).agg(
            Avg_Pred_X=("Avg_Pred_X", "mean"), Avg_Pred_Y=("Avg_Pred_Y", "mean"),
            Num_Sessions=("Session", "nunique"),
        )
        for scope, table in (("All_Sessions", all_average), ("Top_50_Percent", top_average)):
            table["Distance_True_to_Avg_Pred"] = np.linalg.norm(
                table[["Avg_Pred_X", "Avg_Pred_Y"]].to_numpy()
                - table[["True_X", "True_Y"]].to_numpy(), axis=1,
            )
            table.insert(0, "Run", run_name)
            table.insert(1, "Scope", scope)
            average_tables.append(table)
        top_metrics.insert(1, "Top_50_Rank", np.arange(1, len(top_metrics) + 1))
        top_session_tables.append(top_metrics)

        plot_cross_session_average_2d(
            run_avg_df, all_average,
            f"All-Session Average Predicted Location\n{run_name}",
            os.path.join(root_result_dir, f"{run_name}_all_sessions_average_predicted_location.png"),
            "All-Session Average Prediction",
        )
        plot_cross_session_average_2d(
            top_avg_df, top_average,
            f"Top 50% Sessions Average Predicted Location (n={num_top})\n{run_name}",
            os.path.join(root_result_dir, f"{run_name}_top_50_percent_sessions_average_predicted_location.png"),
            "Top 50% Average Prediction",
        )
    return pd.concat(average_tables, ignore_index=True), pd.concat(top_session_tables, ignore_index=True)


def process_session(animal, session_num, saccade_direction, data_dir, args, mapping, root_result_dir, parameter_row):
    session_start_time = time.time()
    session_key = f"{animal}_{saccade_direction}_{session_num}"
    direction_data_dir = data_dir
    print(f"\n================ Processing {animal} {saccade_direction} Session: {session_num} ================\n")
    result_dir = os.path.join(root_result_dir, animal, saccade_direction, session_num)
    os.makedirs(result_dir, exist_ok=True)
    sdf_time_window = slice(args.time_start, args.time_end)

    unit_ids, fixation_files, peri_files = common_condition_units(direction_data_dir, session_num)
    print(f"Common fixation/peri units ({len(unit_ids)}): {unit_ids}")
    fixX, fixY, fix_condition_labels = load_condition_features(
        fixation_files, unit_ids, "fixation", mapping, sdf_time_window,
        args.bin_size_samples, args.kernel_std_samples,
    )
    periX, periY, peri_condition_labels = load_condition_features(
        peri_files, unit_ids, "peri_saccade", mapping, sdf_time_window,
        args.bin_size_samples, args.kernel_std_samples,
    )

    if parameter_row is None:
        print(f"No saved parameters for {session_key}; searching shared X/Y alphas.")
        fix_loo, peri_loo, shared_tuning_df = tune_conditions_shared_independent_xy_no_grid(
            fixX, fixY, periX, periY, args,
            "fixation_train_fixation_test_LOO", "peri_saccade_train_peri_saccade_test_LOO",
            summarize_predictions,
        )
        shared_alpha_x = float(fix_loo["alpha_x"])
        shared_alpha_y = float(fix_loo["alpha_y"])
        parameter_source = "automatic_search_missing_parameter"
        shared_tuning_df["Source"] = parameter_source
        shared_tuning_df["Search_Performed"] = True
    else:
        shared_alpha_x = float(parameter_row["Shared_Alpha_X"])
        shared_alpha_y = float(parameter_row["Shared_Alpha_Y"])
        parameter_source = parameter_row.get("Parameter_Source", "loaded_parameter")
        fixed_alphas = (shared_alpha_x, shared_alpha_y)
        fix_loo = tune_condition_loo_independent_xy_no_grid(
            fixX, fixY, args, "fixation_train_fixation_test_LOO",
            summarize_predictions, fixed_alphas=fixed_alphas,
        )
        peri_loo = tune_condition_loo_independent_xy_no_grid(
            periX, periY, args, "peri_saccade_train_peri_saccade_test_LOO",
            summarize_predictions, fixed_alphas=fixed_alphas,
        )
        shared_tuning_df = pd.DataFrame([{
            "Shared_Alpha_X": shared_alpha_x, "Shared_Alpha_Y": shared_alpha_y,
            "Source": parameter_source,
            "Leftward_Mean_Source_Sessions": parameter_row.get(
                "Leftward_Mean_Source_Sessions", np.nan
            ),
            "Search_Performed": False,
            "Config_Mismatch_Overridden": bool(
                parameter_row.get("Config_Mismatch_Overridden", False)
            ),
            "Config_Mismatch_Details": parameter_row.get(
                "Config_Mismatch_Details", ""
            ),
        }])
    fix_train_peri_test = cross_condition_decode_independent_xy_no_grid(
        fixX, fixY, periX, periY, fix_loo, args,
        "fixation_train_peri_saccade_test", summarize_predictions,
    )
    peri_train_fix_test = cross_condition_decode_independent_xy_no_grid(
        periX, periY, fixX, fixY, peri_loo, args,
        "peri_saccade_train_fixation_test", summarize_predictions,
    )
    runs = [fix_loo, peri_loo, fix_train_peri_test, peri_train_fix_test]
    for run in runs:
        run["metrics"]["Shared_Alpha_X"] = fix_loo["alpha_x"]
        run["metrics"]["Shared_Alpha_Y"] = fix_loo["alpha_y"]
        run["metrics"]["Selected_Alpha_X"] = fix_loo["alpha_x"]
        run["metrics"]["Selected_Alpha_Y"] = fix_loo["alpha_y"]
        run["metrics"]["Parameter_Source"] = parameter_source
        run["metrics"]["Parameter_Sharing"] = "same_alpha_for_fixation_and_peri_saccade_per_axis"
    mismatch_overridden = bool(
        parameter_row is not None
        and parameter_row.get("Config_Mismatch_Overridden", False)
    )
    mismatch_details = (
        parameter_row.get("Config_Mismatch_Details", "")
        if parameter_row is not None else ""
    )
    hparams = {
        "Animal": animal, "Session": session_key, "Raw_Session": session_num, "Saccade_Direction": saccade_direction,
        "Parameter_Config_Mismatch_Overridden": mismatch_overridden,
        "Parameter_Config_Mismatch_Details": mismatch_details,
        "model": "PCA + independent X/Y GPR with shared condition parameters",
        "kernel": "ConstantKernel*RBF", "length_scale_initial": args.length_scale,
        "length_scale_bounds": "(0.1, 10)", "normalize_y": True,
        "grid_projection": False,
        "within_condition_cv": "leave-one-out",
        "within_condition_cv_folds": "n_trials",
        "pca_components": args.pca_components, "kernel_std_samples": args.kernel_std_samples,
        "sdf_time_start": args.time_start, "sdf_time_stop": args.time_end,
        "sdf_bin_size_samples": args.bin_size_samples, "Num_Units": len(unit_ids),
        "Unit_IDs": ",".join(f"{channel}_{unit}" for channel, unit in unit_ids),
        "Fixation_Num_Samples": len(fixY), "Peri_Saccade_Num_Samples": len(periY),
    }
    metrics_df = pd.DataFrame([{**hparams, **run["metrics"]} for run in runs])
    prediction_df = pd.concat([run["prediction_df"] for run in runs], ignore_index=True)
    prediction_df.insert(0, "Saccade_Direction", saccade_direction)
    prediction_df.insert(0, "Raw_Session", str(session_num))
    prediction_df.insert(0, "Session", session_key)
    prediction_df.insert(0, "Animal", animal)
    fold_df = pd.concat([fix_loo["fold_df"], peri_loo["fold_df"]], ignore_index=True)
    fold_df = fold_df.assign(Animal=animal, Session=session_key, Raw_Session=session_num,
                             Saccade_Direction=saccade_direction)
    tuning_df = shared_tuning_df.assign(Animal=animal, Session=session_key, Raw_Session=session_num,
                                        Saccade_Direction=saccade_direction)
    avg_pred_df = pd.concat([
        avg_pred_locs_to_df(*get_avg_pred_locs(run["y_true"], run["y_pred"]), run["name"])
        for run in runs
    ], ignore_index=True).assign(Animal=animal, Session=session_key, Raw_Session=session_num,
                                 Saccade_Direction=saccade_direction)
    report_path = os.path.join(
        result_dir, f"{session_num}_gpr_sdf_pca_cross_condition_independent_xy_loo_aware_unaware_report.xlsx"
    )
    with pd.ExcelWriter(report_path) as writer:
        metrics_df.to_excel(writer, sheet_name="Performance_Metrics", index=False)
        tuning_df.to_excel(writer, sheet_name="Loaded_Parameters", index=False)
        fold_df.to_excel(writer, sheet_name="LOO_Fold_Metrics", index=False)
        prediction_df.to_excel(writer, sheet_name="Predictions", index=False)
        avg_pred_df.to_excel(writer, sheet_name="Avg_Pred_Locations", index=False)
        pd.DataFrame({
            "Condition": ["fixation"] * len(fix_condition_labels) + ["peri_saccade"] * len(peri_condition_labels),
            "Condition_Label": np.concatenate([fix_condition_labels, peri_condition_labels]),
        }).to_excel(writer, sheet_name="Condition_Labels", index=False)
    print(f"Session {session_num} saved to {result_dir}; time={time.time() - session_start_time:.2f}s")
    return metrics_df, tuning_df, avg_pred_df, prediction_df


def main():
    parser = argparse.ArgumentParser(
        description="Run one combined Thor/Ozzy cross-condition analysis."
    )
    parser.add_argument("--session", default=None,
                        help="Backward-compatible session filter applied to both animals.")
    parser.add_argument("--sessions", default="",
                        help="Comma-separated session filter applied to both animals.")
    parser.add_argument("--exclude-sessions", default="",
                        help="Comma-separated session IDs excluded from both animals.")
    parser.add_argument("--animals", default="Thor,Ozzy",
                        help="Animals to include: Thor,Ozzy (default: both).")
    parser.add_argument(
        "--thor-base-data-dir", default=r"C:\Data\spike_probe_misloc_thor_filt_trials",
        help="Thor session root. Thor is always treated as rightward.",
    )
    parser.add_argument(
        "--ozzy-base-data-dir", "--base-data-dir", dest="ozzy_base_data_dir",
        default=r"C:\Data\spike_probe_misloc_ozzy_filt_trials_all",
        help="Ozzy root containing left and right directories.",
    )
    parser.add_argument(
        "--ozzy-saccade-directions", "--saccade-directions",
        dest="ozzy_saccade_directions", default="left,right",
        help="Ozzy directions to include. This option never changes Thor from rightward.",
    )
    parser.add_argument(
        "--time-start", type=int, default=60,
        help="Common analysis and parameter-window start (default: 60).",
    )
    parser.add_argument(
        "--time-end", type=int, default=140,
        help="Common analysis and parameter-window end (default: 140).",
    )
    parser.add_argument("--bin-size-samples", type=int, default=10)
    parser.add_argument(
        "--kernel-std-samples", type=float, default=10.0,
        help=(
            "Gaussian SDF kernel standard deviation used by both parameter "
            "search and cross-test (default: 10 samples)."
        ),
    )
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--thor-parameter-file", default="",
                        help="Thor shared-alpha CSV; empty uses the Thor default path.")
    parser.add_argument(
        "--ozzy-parameter-file", "--parameter-file", dest="ozzy_parameter_file", default="",
        help="Ozzy shared-alpha CSV; empty uses the Ozzy default path.",
    )
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="Initial RBF kernel length scale.")
    parser.add_argument("--alpha-candidates", default="1e-2,1e-1,2e-1,4e-1,6e-1,8e-1")
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    # None makes gpr_cross_pca_utils use sklearn LeaveOneOut.
    args.tuning_cv_folds = None

    args.alpha_candidates = [
        float(value) for value in parse_name_list(args.alpha_candidates)
    ]
    if not args.alpha_candidates or any(value <= 0 for value in args.alpha_candidates):
        parser.error("--alpha-candidates must contain positive values.")
    try:
        animals = normalize_animals(args.animals)
        ozzy_directions = normalize_directions(args.ozzy_saccade_directions)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pca_components < 1:
        parser.error("--pca-components must be positive.")
    if args.time_end <= args.time_start or (
            args.time_end - args.time_start) % args.bin_size_samples:
        parser.error("The time window must be positive and divisible by --bin-size-samples.")

    mapping = {
        56: [0, 2], 55: [1, 2], 54: [2, 2],
        57: [0, 1], 51: [1, 1], 53: [2, 1],
        58: [0, 0], 59: [1, 0], 52: [2, 0],
    }
    total_start_time = time.time()
    requested = (
        [args.session] if args.session else parse_name_list(args.sessions)
    )
    excluded = parse_name_list(args.exclude_sessions)
    analyses = discover_combined_sessions(
        discover_cross_sessions,
        animals,
        ozzy_directions,
        args.thor_base_data_dir,
        args.ozzy_base_data_dir,
        requested,
        excluded,
    )
    print(
        "Combined datasets: "
        + ", ".join(
            f"{animal}={sum(item['animal'] == animal for item in analyses)}"
            for animal in animals
        )
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.thor_parameter_file:
        args.thor_parameter_file = os.path.join(
            script_dir,
            default_parameter_file(
                "Thor",
                args.time_start,
                args.time_end,
                args.bin_size_samples,
            ),
        )
    if not args.ozzy_parameter_file:
        args.ozzy_parameter_file = os.path.join(
            script_dir,
            default_parameter_file(
                "Ozzy",
                args.time_start,
                args.time_end,
                args.bin_size_samples,
            ),
        )
    parameter_context = load_parameter_context(
        args.thor_parameter_file, args.ozzy_parameter_file
    )

    root_result_dir = os.path.join(
        script_dir,
        "Results",
        "fix_peri_cross_test",
    )
    os.makedirs(root_result_dir, exist_ok=True)
    all_metrics, all_tuning = [], []
    all_session_avg, all_trial_predictions, failures = [], [], []
    for item in analyses:
        animal = item["animal"]
        direction = item["direction"]
        session = item["session"]
        try:
            parameter_row = find_parameter_row(
                parameter_context, animal, direction, session
            )
            expected_config = {
                "PCA_Components": args.pca_components,
                "Length_Scale_Initial": args.length_scale,
                "Kernel_Std_Samples": args.kernel_std_samples,
                "Time_Start": args.time_start,
                "Time_End": args.time_end,
                "Bin_Size_Samples": args.bin_size_samples,
            }
            mismatches = [
                f"{column}: saved={parameter_row[column]}, current={expected}"
                for column, expected in expected_config.items()
                if parameter_row is not None
                and column in parameter_row.index
                and pd.notna(parameter_row[column])
                and not np.isclose(float(parameter_row[column]), float(expected))
            ]
            if mismatches:
                mismatch_details = "; ".join(mismatches)
                print(
                    f"WARNING: {animal} {direction} session {session} parameter "
                    f"config mismatch overridden; loading saved alpha(s) with the "
                    f"current analysis window {args.time_start}:{args.time_end}. "
                    f"Details: {mismatch_details}"
                )
                parameter_row = parameter_row.copy()
                parameter_row["Config_Mismatch_Overridden"] = True
                parameter_row["Config_Mismatch_Details"] = mismatch_details
            metrics, tuning, session_avg, trial_predictions = process_session(
                animal, session, direction, item["data_dir"], args, mapping,
                root_result_dir, parameter_row,
            )
            all_metrics.append(metrics)
            all_tuning.append(tuning)
            all_session_avg.append(session_avg)
            all_trial_predictions.append(trial_predictions)
        except Exception as exc:
            identity = f"{animal}_{direction}_{session}"
            print(f"{identity} FAILED: {exc}")
            failures.append({
                "Animal": animal,
                "Session": identity,
                "Raw_Session": session,
                "Saccade_Direction": direction,
                "Error": str(exc),
            })

    if not all_metrics:
        raise RuntimeError(
            "No combined Thor/Ozzy analysis completed successfully; inspect Failed_Sessions."
        )
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    session_avg_df = pd.concat(all_session_avg, ignore_index=True)
    trial_predictions_df = pd.concat(all_trial_predictions, ignore_index=True)
    (
        aware_unaware_scatter_df,
        aware_unaware_statistics_df,
        top50_aware_unaware_summary_df,
    ) = create_aware_vs_unaware_signed_scatter(
        metrics_df, session_avg_df, trial_predictions_df, root_result_dir
    )
    summary_path = os.path.join(root_result_dir, "all_sessions_p6_loo_aware_unaware_thor_ozzy_summary.xlsx")
    with pd.ExcelWriter(summary_path) as writer:
        metrics_df.to_excel(writer, sheet_name="Performance_Metrics", index=False)
        pd.concat(all_tuning, ignore_index=True).to_excel(
            writer, sheet_name="Loaded_Parameters", index=False
        )
        session_avg_df.to_excel(writer, sheet_name="Session_Avg_Locations", index=False)

        aware_unaware_scatter_df.to_excel(
            writer, sheet_name="Aware_Unaware_Scatter", index=False
        )
        aware_unaware_statistics_df.to_excel(
            writer, sheet_name="Aware_Unaware_Statistics", index=False
        )
        pd.DataFrame(failures).to_excel(writer, sheet_name="Failed_Sessions", index=False)

    top50_statistics_path = os.path.join(
        root_result_dir, "top_50_percent_aware_unaware_statistical_summary.xlsx"
    )
    top50_definitions_df = pd.DataFrame([
        {
            "Field": "Selection",
            "Definition": (
                "Each monkey's top 50% probes by raw fixation error are selected "
                "separately, then combined; sessions retain at least 3 selected probes."
            ),
        },
        {
            "Field": "Independent_Unit",
            "Definition": (
                "Animal + Session + Saccade_Direction mean across eligible probe points."
            ),
        },
        {
            "Field": "IQR_Width",
            "Definition": "Q3 - Q1 of independent-unit signed delta values.",
        },
        {
            "Field": "Condition_vs_Zero_Wilcoxon_P",
            "Definition": "Two-sided one-sample Wilcoxon signed-rank p-value versus zero.",
        },
        {
            "Field": "Aware_vs_Unaware_Wilcoxon_P",
            "Definition": (
                "Two-sided paired Wilcoxon signed-rank p-value for session-level "
                "Aware minus Unaware signed delta."
            ),
        },
    ])
    with pd.ExcelWriter(top50_statistics_path) as writer:
        top50_aware_unaware_summary_df.to_excel(
            writer, sheet_name="Top50_Summary", index=False
        )
        top50_definitions_df.to_excel(writer, sheet_name="Definitions", index=False)

    print(
        f"Completed {len(all_metrics)}/{len(analyses)} combined datasets in "
        f"{(time.time() - total_start_time) / 60:.2f} min; summary={summary_path}; "
        f"top50_statistics={top50_statistics_path}"
    )


if __name__ == "__main__":
    main()
