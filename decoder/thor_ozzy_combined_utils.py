import os

import numpy as np
import pandas as pd


VALID_ANIMALS = {"Thor", "Ozzy"}
VALID_DIRECTIONS = {"left", "right"}


def parse_name_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_animals(value):
    aliases = {"thor": "Thor", "ozzy": "Ozzy"}
    animals = []
    invalid = []
    for item in parse_name_list(value):
        animal = aliases.get(item.lower())
        if animal is None:
            invalid.append(item)
        elif animal not in animals:
            animals.append(animal)
    if invalid or not animals:
        raise ValueError(f"--animals must contain Thor and/or Ozzy; invalid={invalid}")
    return animals


def normalize_directions(value):
    directions = []
    invalid = []
    for item in parse_name_list(value):
        direction = item.lower()
        if direction not in VALID_DIRECTIONS:
            invalid.append(item)
        elif direction not in directions:
            directions.append(direction)
    if invalid or not directions:
        raise ValueError(
            f"--ozzy-saccade-directions must contain left and/or right; invalid={invalid}"
        )
    return directions


def discover_combined_sessions(
        discover_cross_sessions, animals, ozzy_directions, thor_base_data_dir,
        ozzy_base_data_dir, requested=None, excluded=None):
    analyses = []
    if "Thor" in animals:
        if not os.path.isdir(thor_base_data_dir):
            raise FileNotFoundError(f"Thor data directory not found: {thor_base_data_dir}")
        sessions = discover_cross_sessions(
            thor_base_data_dir, requested or None, excluded or None
        )
        analyses.extend({
            "animal": "Thor",
            "direction": "right",
            "session": session,
            "data_dir": thor_base_data_dir,
        } for session in sessions)

    if "Ozzy" in animals:
        for direction in ozzy_directions:
            data_dir = os.path.join(ozzy_base_data_dir, direction)
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"Ozzy {direction} data directory not found: {data_dir}")
            sessions = discover_cross_sessions(data_dir, requested or None, excluded or None)
            analyses.extend({
                "animal": "Ozzy",
                "direction": direction,
                "session": session,
                "data_dir": data_dir,
            } for session in sessions)

    identities = [
        (item["animal"], item["direction"], str(item["session"])) for item in analyses
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate animal/direction/session datasets were discovered.")
    if not analyses:
        raise RuntimeError("No Thor/Ozzy sessions containing both conditions were found.")
    return analyses


def default_parameter_file(animal, time_start, time_end, bin_size_samples):
    search_name = (
        "fix_peri_cross_test_shared_alpha_parameter_search_ozzy"
        if animal == "Ozzy"
        else "fix_peri_cross_test_shared_alpha_parameter_search"
    )
    return os.path.join(
        "Results",
        f"sdf_binned_{time_start}_{time_end}_{bin_size_samples}ms",
        search_name,
        "shared_alpha_parameters.csv",
    )


def _load_parameter_table(path, animal):
    required = {"Session", "Shared_Alpha_X", "Shared_Alpha_Y"}
    if not os.path.isfile(path):
        print(f"{animal} parameter file not found ({path}); missing sessions will tune automatically.")
        return pd.DataFrame({
            "Session": pd.Series(dtype=str),
            "Shared_Alpha_X": pd.Series(dtype=float),
            "Shared_Alpha_Y": pd.Series(dtype=float),
        })
    table = pd.read_csv(path, dtype={"Session": str})
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{animal} parameter file is missing columns: {sorted(missing)}")
    table = table.copy()
    table["Session"] = table["Session"].astype(str)
    table["Shared_Alpha_X"] = pd.to_numeric(table["Shared_Alpha_X"], errors="coerce")
    table["Shared_Alpha_Y"] = pd.to_numeric(table["Shared_Alpha_Y"], errors="coerce")
    if "Saccade_Direction" in table.columns:
        table["Saccade_Direction"] = table["Saccade_Direction"].astype(str).str.lower()
        duplicates = table.duplicated(["Saccade_Direction", "Session"])
    else:
        duplicates = table.duplicated(["Session"])
    if duplicates.any():
        raise ValueError(
            f"{animal} parameter file contains duplicate session rows: "
            f"{table.loc[duplicates, 'Session'].tolist()}"
        )
    table["Parameter_Source"] = path
    return table


def load_parameter_context(thor_path, ozzy_path):
    tables = {
        "Thor": _load_parameter_table(thor_path, "Thor"),
        "Ozzy": _load_parameter_table(ozzy_path, "Ozzy"),
    }
    ozzy = tables["Ozzy"]
    if "Saccade_Direction" in ozzy.columns:
        left_rows = ozzy[ozzy["Saccade_Direction"].eq("left")]
    else:
        left_rows = ozzy
    valid = left_rows[
        np.isfinite(left_rows["Shared_Alpha_X"])
        & np.isfinite(left_rows["Shared_Alpha_Y"])
        & (left_rows["Shared_Alpha_X"] > 0)
        & (left_rows["Shared_Alpha_Y"] > 0)
    ]
    ozzy_left_mean = None
    if not valid.empty:
        ozzy_left_mean = {
            "Shared_Alpha_X": float(valid["Shared_Alpha_X"].mean()),
            "Shared_Alpha_Y": float(valid["Shared_Alpha_Y"].mean()),
            "Leftward_Mean_Source_Sessions": int(len(valid)),
            "Parameter_Source": "ozzy_leftward_parameter_mean_fallback",
        }
    return {"tables": tables, "ozzy_left_mean": ozzy_left_mean}


def find_parameter_row(context, animal, direction, session):
    table = context["tables"][animal]
    matches = table[table["Session"].eq(str(session))]
    if "Saccade_Direction" in table.columns:
        matches = matches[matches["Saccade_Direction"].eq(direction)]
    elif animal == "Ozzy" and direction != "left":
        # Legacy Ozzy tables contain leftward sessions only.
        matches = matches.iloc[0:0]
    if not matches.empty:
        return matches.iloc[0].copy()
    if animal == "Ozzy" and direction == "right" and context["ozzy_left_mean"]:
        return pd.Series(context["ozzy_left_mean"])
    return None
