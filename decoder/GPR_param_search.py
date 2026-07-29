import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gpr_cross_pca_utils import (
    common_condition_units,
    discover_cross_sessions,
    load_condition_features,
    tune_conditions_shared_independent_xy_no_grid,
)
from thor_ozzy_combined_utils import (
    discover_combined_sessions,
    normalize_animals,
    normalize_directions,
    parse_name_list,
)


def parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def safe_pearson_avg(y_true, y_pred):
    correlations = []
    for axis_index in range(y_true.shape[1]):
        if (
            len(y_true) < 2
            or np.std(y_true[:, axis_index]) == 0
            or np.std(y_pred[:, axis_index]) == 0
        ):
            correlations.append(np.nan)
        else:
            correlations.append(
                pearsonr(y_true[:, axis_index], y_pred[:, axis_index])[0]
            )
    return float(np.nanmean(correlations))


def summarize_predictions(y_true, y_pred, run_name):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "Run": run_name,
        "Num_Test_Samples": len(y_true),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_avg": safe_pearson_avg(y_true, y_pred),
        "Mean_Euclidean_Error": float(
            np.mean(np.linalg.norm(y_pred - y_true, axis=1))
        ),
    }


def animal_tuning_config(args, animal):
    """Return the common preprocessing configuration used for both monkeys."""
    if animal not in ("Thor", "Ozzy"):
        raise ValueError(f"Unsupported animal: {animal}")
    return {
        "time_start": args.time_start,
        "time_end": args.time_end,
        "kernel_std_samples": args.kernel_std_samples,
    }


def parameter_output_dir(output_root, animal, config, bin_size_samples):
    folder_name = (
        "fix_peri_cross_test_shared_alpha_parameter_search"
        if animal == "Thor"
        else "fix_peri_cross_test_shared_alpha_parameter_search_ozzy"
    )
    return os.path.join(
        output_root,
        (
            f"sdf_binned_{config['time_start']}_{config['time_end']}_"
            f"{bin_size_samples}ms"
        ),
        folder_name,
    )


def search_dataset_parameters(dataset, args, mapping):
    animal = dataset["animal"]
    direction = dataset["direction"]
    session = str(dataset["session"])
    config = animal_tuning_config(args, animal)

    tuning_args = copy.copy(args)
    tuning_args.time_start = config["time_start"]
    tuning_args.time_end = config["time_end"]
    tuning_args.kernel_std_samples = config["kernel_std_samples"]

    units, fixation_files, peri_files = common_condition_units(
        dataset["data_dir"], session
    )
    time_window = slice(tuning_args.time_start, tuning_args.time_end)
    fix_x, fix_y, _ = load_condition_features(
        fixation_files,
        units,
        "fixation",
        mapping,
        time_window,
        tuning_args.bin_size_samples,
        tuning_args.kernel_std_samples,
    )
    peri_x, peri_y, _ = load_condition_features(
        peri_files,
        units,
        "peri_saccade",
        mapping,
        time_window,
        tuning_args.bin_size_samples,
        tuning_args.kernel_std_samples,
    )
    fix_result, _, tuning_df = tune_conditions_shared_independent_xy_no_grid(
        fix_x,
        fix_y,
        peri_x,
        peri_y,
        tuning_args,
        "fixation_train_fixation_test_LOO",
        "peri_saccade_train_peri_saccade_test_LOO",
        summarize_predictions,
    )

    parameter_row = {
        "Animal": animal,
        "Saccade_Direction": direction,
        "Session": session,
        "Shared_Alpha_X": fix_result["alpha_x"],
        "Shared_Alpha_Y": fix_result["alpha_y"],
        "Alpha_Candidates": ",".join(
            f"{value:g}" for value in tuning_args.alpha_candidates
        ),
        "PCA_Components": tuning_args.pca_components,
        "Length_Scale_Initial": tuning_args.length_scale,
        "Length_Scale_Bounds": "0.1,10.0",
        "Kernel_Std_Samples": tuning_args.kernel_std_samples,
        "Time_Start": tuning_args.time_start,
        "Time_End": tuning_args.time_end,
        "Bin_Size_Samples": tuning_args.bin_size_samples,
        "Num_Units": len(units),
        "Parameter_Sharing": (
            "same_alpha_for_fixation_and_peri_saccade_per_axis"
        ),
        "Tuning_CV": (
            "leave_one_out"
            if tuning_args.tuning_cv_folds is None
            else f"{tuning_args.tuning_cv_folds}_fold"
        ),
    }
    tuning_df = tuning_df.copy()
    tuning_df.insert(0, "Session", session)
    tuning_df.insert(0, "Saccade_Direction", direction)
    tuning_df.insert(0, "Animal", animal)
    print(
        f"{animal} {direction} session {session}: "
        f"Shared_Alpha_X={fix_result['alpha_x']:g}, "
        f"Shared_Alpha_Y={fix_result['alpha_y']:g}"
    )
    return parameter_row, tuning_df


def write_parameter_outputs(
    parameter_df, tuning_df, failures_df, output_root, args
):
    output_rows = []
    for animal in ("Thor", "Ozzy"):
        animal_parameters = parameter_df[parameter_df["Animal"].eq(animal)]
        if animal_parameters.empty:
            continue
        config = animal_tuning_config(args, animal)
        output_dir = parameter_output_dir(
            output_root, animal, config, args.bin_size_samples
        )
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "shared_alpha_parameters.csv")
        report_path = os.path.join(
            output_dir, "shared_alpha_parameter_search_report.xlsx"
        )
        animal_tuning = tuning_df[tuning_df["Animal"].eq(animal)]
        animal_failures = failures_df[failures_df["Animal"].eq(animal)]
        animal_parameters.to_csv(csv_path, index=False)
        with pd.ExcelWriter(report_path) as writer:
            animal_parameters.to_excel(
                writer, sheet_name="Selected_Parameters", index=False
            )
            animal_tuning.to_excel(
                writer, sheet_name="Parameter_Tuning", index=False
            )
            animal_failures.to_excel(
                writer, sheet_name="Failed_Sessions", index=False
            )
        output_rows.append(
            {
                "Animal": animal,
                "Time_Start": config["time_start"],
                "Time_End": config["time_end"],
                "Kernel_Std_Samples": config["kernel_std_samples"],
                "Parameter_CSV": os.path.abspath(csv_path),
                "Report_XLSX": os.path.abspath(report_path),
            }
        )

    combined_dir = os.path.join(
        output_root, "Thor_Ozzy_combined_parameter_search"
    )
    os.makedirs(combined_dir, exist_ok=True)
    combined_csv = os.path.join(
        combined_dir, "combined_shared_alpha_parameters.csv"
    )
    combined_report = os.path.join(
        combined_dir, "combined_shared_alpha_parameter_search_report.xlsx"
    )
    parameter_df.to_csv(combined_csv, index=False)
    with pd.ExcelWriter(combined_report) as writer:
        parameter_df.to_excel(
            writer, sheet_name="Selected_Parameters", index=False
        )
        tuning_df.to_excel(writer, sheet_name="Parameter_Tuning", index=False)
        failures_df.to_excel(writer, sheet_name="Failed_Sessions", index=False)
        pd.DataFrame(output_rows).to_excel(
            writer, sheet_name="Output_Files", index=False
        )
    return output_rows, combined_csv, combined_report


def validate_window(parser, label, time_start, time_end, bin_size_samples):
    if time_end <= time_start or (time_end - time_start) % bin_size_samples:
        parser.error(
            f"{label} time window must be positive and divisible by "
            "--bin-size-samples."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune session-wise shared X/Y GPR alpha parameters for Thor and "
            "Ozzy in one run. Thor is always rightward; Ozzy can include "
            "left and/or right datasets."
        )
    )
    parser.add_argument(
        "--animals", default="Thor,Ozzy",
        help="Animals to tune: Thor,Ozzy (default: both).",
    )
    parser.add_argument(
        "--session", default=None,
        help="Backward-compatible session filter applied to both animals.",
    )
    parser.add_argument(
        "--sessions", default="",
        help="Comma-separated session filter applied to both animals.",
    )
    parser.add_argument(
        "--exclude-sessions", default="",
        help="Comma-separated session IDs excluded from both animals.",
    )
    parser.add_argument(
        "--thor-base-data-dir",
        default=r"C:\Data\spike_probe_misloc_thor_filt_trials",
    )
    parser.add_argument(
        "--ozzy-base-data-dir",
        default=r"C:\Data\spike_probe_misloc_ozzy_filt_trials_all",
        help="Ozzy root containing left and right subdirectories.",
    )
    parser.add_argument(
        "--ozzy-saccade-directions",
        default="left,right",
        help="Ozzy directions to tune. Thor remains rightward.",
    )
    parser.add_argument(
        "--time-start", type=int, default=60,
        help="Common analysis-window start for both monkeys (default: 60).",
    )
    parser.add_argument(
        "--time-end", type=int, default=140,
        help="Common analysis-window end for both monkeys (default: 140).",
    )
    parser.add_argument(
        "--kernel-std-samples", type=float, default=10.0,
        help=(
            "Common Gaussian SDF kernel standard deviation in samples for "
            "both monkeys (default: 10)."
        ),
    )
    parser.add_argument("--bin-size-samples", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument(
        "--alpha-candidates",
        default="1e-2,1e-1,2e-1,4e-1,6e-1,8e-1",
    )
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tuning-cv-folds",
        type=int,
        default=None,
        help="Default is leave-one-out; provide an integer for K-fold tuning.",
    )
    parser.add_argument(
        "--standardize-features", action="store_true",
        help="Standardize features within each CV fold before PCA.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "Results"
        ),
        help=(
            "Root directory for parameter CSVs and reports "
            "(default: the script's Results directory)."
        ),
    )
    args = parser.parse_args()

    try:
        animals = normalize_animals(args.animals)
        ozzy_directions = normalize_directions(args.ozzy_saccade_directions)
    except ValueError as exc:
        parser.error(str(exc))
    args.alpha_candidates = parse_float_list(args.alpha_candidates)
    if not args.alpha_candidates or any(
        value <= 0 for value in args.alpha_candidates
    ):
        parser.error("--alpha-candidates must contain positive values.")
    if args.pca_components < 1:
        parser.error("--pca-components must be positive.")
    if args.bin_size_samples < 1:
        parser.error("--bin-size-samples must be positive.")
    if args.kernel_std_samples <= 0:
        parser.error("--kernel-std-samples must be positive.")
    if args.tuning_cv_folds is not None and args.tuning_cv_folds < 2:
        parser.error("--tuning-cv-folds must be at least 2.")
    validate_window(
        parser,
        "Common Thor/Ozzy",
        args.time_start,
        args.time_end,
        args.bin_size_samples,
    )

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
        "Parameter-search datasets: "
        + ", ".join(
            f"{animal}={sum(item['animal'] == animal for item in analyses)}"
            for animal in animals
        )
    )

    mapping = {
        56: [0, 2], 55: [1, 2], 54: [2, 2],
        57: [0, 1], 51: [1, 1], 53: [2, 1],
        58: [0, 0], 59: [1, 0], 52: [2, 0],
    }
    start_time = time.time()
    parameter_rows, tuning_tables, failures = [], [], []
    for dataset in analyses:
        try:
            parameter_row, tuning_df = search_dataset_parameters(
                dataset, args, mapping
            )
            parameter_rows.append(parameter_row)
            tuning_tables.append(tuning_df)
        except Exception as exc:
            print(
                f"{dataset['animal']} {dataset['direction']} session "
                f"{dataset['session']} FAILED: {exc}"
            )
            failures.append(
                {
                    "Animal": dataset["animal"],
                    "Saccade_Direction": dataset["direction"],
                    "Session": str(dataset["session"]),
                    "Data_Directory": dataset["data_dir"],
                    "Error": str(exc),
                }
            )
    if not parameter_rows:
        raise RuntimeError(
            "No Thor/Ozzy parameter search completed successfully."
        )

    parameter_df = pd.DataFrame(parameter_rows).sort_values(
        ["Animal", "Saccade_Direction", "Session"]
    )
    tuning_df = pd.concat(tuning_tables, ignore_index=True)
    failures_df = pd.DataFrame(
        failures,
        columns=[
            "Animal", "Saccade_Direction", "Session",
            "Data_Directory", "Error",
        ],
    )
    output_rows, combined_csv, combined_report = write_parameter_outputs(
        parameter_df, tuning_df, failures_df, args.output_root, args
    )
    print(
        f"Completed {len(parameter_rows)}/{len(analyses)} parameter searches "
        f"in {(time.time() - start_time) / 60:.2f} min."
    )
    for row in output_rows:
        print(f"{row['Animal']} parameters: {row['Parameter_CSV']}")
    print(f"Combined parameters: {os.path.abspath(combined_csv)}")
    print(f"Combined report: {os.path.abspath(combined_report)}")


if __name__ == "__main__":
    main()
