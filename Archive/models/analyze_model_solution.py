from utils.header import *

SEG_GROUPS = {
    "num_black_to_school": "perblk",
    "num_asian_to_school": "perasn",
    "num_native_to_school": "pernam",
    "num_hispanic_to_school": "perhsp",
    "num_ell_to_school": "perell",
    "num_frl_to_school": "perfrl",
}

CATS = {
    "hisp": ("num_hispanic_to_school", "perhsp"),
    "black": ("num_black_to_school", "perblk"),
    "ell": ("num_ell_to_school", "perell"),
    "frl": ("num_frl_to_school", "perfrl"),
    "white": ("num_white_to_school", "perwht"),
    "asian": ("num_asian_to_school", "perasn"),
    "native": ("num_native_to_school", "pernam"),
}


def identify_switched_blocks(
    district_id,
    year,
    state,
    solution_file,
    max_percent_distance_increase,
    max_percent_size_increase,
    optimizing_for,
    objective_function,
    percent_neighbors_rezoned_together,
    enforce_contiguity,
    pre_solver_file="models/solver_files/{}/{}/prepped_file_for_solver_{}.csv",
    solution_dir="models/results/{}/{}/{}/{}_{}_{}_{}_{}_{}/",
    input_file_schools="data/derived_data/{}/{}/schools_file_for_assignment.csv",
):

    try:
        df_orig = pd.read_csv(pre_solver_file.format(year, state, district_id)).rename(
            columns={"ncessch": "orig_nces"}
        )
        df_soln = pd.read_csv(
            solution_dir.format(
                year,
                state,
                district_id,
                max_percent_distance_increase,
                max_percent_size_increase,
                optimizing_for,
                objective_function,
                percent_neighbors_rezoned_together,
                enforce_contiguity,
            )
            + solution_file
        )
        df_soln = pd.merge(df_soln, df_orig, on="block_id", how="inner")
    except Exception as e:
        return

    switched_blocks = []
    for i in range(0, len(df_soln)):
        if df_soln["new_school_nces"][i] != df_soln["orig_nces"][i]:
            switched_blocks.append(df_soln["block_id"][i])

    print(
        "Fraction of blocks switched: ",
        len(switched_blocks) / len(df_soln),
    )


def analyze_school_distributions_pre_post(
    district_id,
    year,
    state,
    solution_dir,
    cat_optimizing_for,
    school_nces,
    pre_solver_file="models/solver_files/{}/{}/prepped_file_for_solver_{}.csv",
    output_file="analysis_output/{}_{}.csv",
):
    df_orig = pd.read_csv(
        pre_solver_file.format(year, state, district_id), dtype=str
    ).rename(columns={"ncessch": "orig_nces"})
    df_orig["num_total_to_school"] = df_orig["num_total_to_school"].astype(float)
    df_orig["num_white_to_school"] = df_orig["num_white_to_school"].astype(float)
    df_orig["num_black_to_school"] = df_orig["num_black_to_school"].astype(float)
    df_orig_g = df_orig.groupby("orig_nces", as_index=False).agg(
        {
            "num_total_to_school": "sum",
            "num_white_to_school": "sum",
            "num_black_to_school": "sum",
        }
    )

    df_target_school_orig = df_orig_g[df_orig_g["orig_nces"].isin([school_nces])].iloc[
        0
    ]

    data_out = {
        "pre_white": [],
        "pre_black": [],
        "post_white": [],
        "post_black": [],
        "config": [],
    }
    for config in os.listdir(solution_dir):
        if not cat_optimizing_for in config:
            continue
        solution_file = ""
        for f in os.listdir(solution_dir + "/" + config + "/"):
            if f.startswith("solution"):
                solution_file = f

        if solution_file:
            df_soln = pd.read_csv(
                solution_dir + "/" + config + "/" + solution_file, dtype=str
            )
            df_soln["num_total_to_school"] = df_soln["num_total_to_school"].astype(
                float
            )
            df_soln["num_white_to_school"] = df_soln["num_white_to_school"].astype(
                float
            )
            df_soln["num_black_to_school"] = df_soln["num_black_to_school"].astype(
                float
            )
            df_soln_g = df_soln.groupby("new_school_nces", as_index=False).agg(
                {
                    "num_total_to_school": "sum",
                    "num_white_to_school": "sum",
                    "num_black_to_school": "sum",
                }
            )

            df_target_school_soln = df_soln_g[
                df_soln_g["new_school_nces"].isin([school_nces])
            ].iloc[0]

            data_out["pre_white"].append(
                df_target_school_orig["num_white_to_school"]
                / df_target_school_orig["num_total_to_school"]
            )

            data_out["pre_black"].append(
                df_target_school_orig["num_black_to_school"]
                / df_target_school_orig["num_total_to_school"]
            )

            data_out["post_white"].append(
                df_target_school_soln["num_white_to_school"]
                / df_target_school_soln["num_total_to_school"]
            )

            data_out["post_black"].append(
                df_target_school_soln["num_black_to_school"]
                / df_target_school_soln["num_total_to_school"]
            )

            data_out["config"].append(config)

    df_out = pd.DataFrame(data=data_out)
    df_out.to_csv(output_file.format(school_nces, cat_optimizing_for), index=False)


def get_name_for_solver_status(code):
    code = int(code)
    if code == 0:
        return "unknown"
    elif code == 1:
        return "model_invalid"
    elif code == 2:
        return "feasible"
    elif code == 3:
        return "infeasible"
    elif code == 4:
        return "optimal"


def add_idx_fields(df_orig, df_soln):
    df_orig_g = (
        df_orig.groupby("ncessch", as_index=False)
        .agg({"school_idx": "first"})
        .reset_index(drop=True)
    )

    df_soln_g = (
        df_soln.groupby("new_school_nces", as_index=False)
        .agg({"perfrl": "first"})
        .reset_index(drop=True)
        .drop(columns=["perfrl"])
    )

    df_soln_g = pd.merge(
        df_soln_g, df_orig_g, left_on="new_school_nces", right_on="ncessch", how="left"
    )

    # Add in block idx field to solution
    df_block_idx = df_orig[["block_idx", "block_id"]]
    df = pd.merge(
        df_soln,
        df_block_idx,
        on="block_id",
        how="left",
    )

    # Add in school idx field to solution
    df = pd.merge(df, df_soln_g, on="new_school_nces", how="left")
    return df


def add_contiguous_suffix_to_local_solution_dirs(
    solution_dir="simulation_outputs/va_2122_exposure/{}/{}/",
    state="VA",
    year="2122",
    suffix="_False",
):
    solution_dir = solution_dir.format(year, state)
    for district_id in os.listdir(solution_dir):
        print(district_id)
        for config in os.listdir(solution_dir + district_id + "/"):
            if len(config.split("_")) < 8:
                path = solution_dir + district_id + "/" + config
                os.rename(path, path + suffix)


def remove_solution_dirs(
    solution_dir="simulation_outputs/all_usa_2122/2122/",
    suffix="False",
):
    for state in os.listdir(solution_dir):
        for district_id in os.listdir(os.path.join(solution_dir, state)):
            print(district_id)
            for config in os.listdir(os.path.join(solution_dir, state, district_id)):
                config_params = config.split("_")
                if config_params[7] == suffix:
                    shutil.rmtree(
                        os.path.join(solution_dir, state, district_id, config)
                    )


def filter_and_update_consolidated_sims_file(
    input_file="simulation_outputs/va_2122_exposure/consolidated_simulation_results.csv",
    max_percent_distance_increases=[0.5, 1],
    max_percent_size_increases=[0.1, 0.15, 0.2],
    percent_neighbors_rezoned_together=[0.5],
    optimizing_for=["black", "white", "hisp", "ell", "frl"],
    objective_functions=["min_total"],
    output_file="simulation_outputs/va_2122_exposure/consolidated_simulation_results_filtered.csv",
):
    df = pd.read_csv(input_file)
    df["is_contiguous"] = [False for i in range(0, len(df))]
    df = df[
        df["config"].isin(["status_quo_zoning"])
        | (
            df["travel_time_threshold"].isin(max_percent_distance_increases)
            & df["school_size_threshold"].isin(max_percent_size_increases)
            & df["objective_function"].isin(objective_functions)
            & df["cat_optimizing_for"].isin(optimizing_for)
            & df["community_cohesion_threshold"].isin(
                percent_neighbors_rezoned_together
            )
        )
    ].reset_index(drop=True)
    configs = []
    for i in range(0, len(df)):
        if df["config"][i] == "status_quo_zoning":
            configs.append(df["config"][i])
        else:
            configs.append(df["config"][i] + "_False")

    df["config"] = configs
    print(len(df))
    df.to_csv(output_file, index=False)


def estimate_outcomes_for_choice_scenario(
    key,
    df_orig,
    df_soln,
    choice_multiplier,
    config,
    choice_data="data/school_covariates/school_choice_analysis.csv",
):
    # Since we are focused on white/non-white seg for the paper, only focus on 'White' for now
    # and skip others
    if (
        config == "status_quo_zoning"
        or key == "num_frl_to_school"
        or key == "num_ell_to_school"
        or key == "num_black_to_school"
        or key == "num_hispanic_to_school"
        or key == "num_native_to_school"
        or key == "num_asian_to_school"
    ):
        return float("nan"), float("nan"), float("nan")

    # Just renaming a column to make it easier to write more general code later below
    df_choice = pd.read_csv(choice_data).rename(
        columns={"ratio_c_or_m_to_dist_enroll": "ratio_c_or_m_to_dist_total"}
    )

    # If the current district does not have choice skip this and move on
    if df_orig["leaid"][0] not in df_choice["leaid"].tolist():
        return float("nan"), float("nan"), float("nan")

    df_orig = df_orig.rename(columns={"ncessch": "orig_nces_id"})
    df_soln_subset = df_soln[["block_id", "new_school_nces"]]
    df_combined = pd.merge(df_orig, df_soln_subset, on="block_id", how="inner")
    df_combined["school_changed"] = (
        df_combined["orig_nces_id"] != df_combined["new_school_nces"]
    )

    dist_choice = df_choice[df_choice["leaid"] == df_combined["leaid"][0]].iloc[0]

    # Now, for those blocks that switched schools, apply the opt out rate
    optout_estimates = {
        "num_total_to_school": [],
        "num_white_to_school": [],
        "num_black_to_school": [],
        "num_native_to_school": [],
        "num_hispanic_to_school": [],
        "num_asian_to_school": [],
    }
    keys_to_keep = [
        "block_id",
        "new_school_nces",
        "district_perasn",
        "district_perblk",
        "district_perell",
        "district_perfrl",
        "district_perhsp",
        "district_pernam",
        "district_perwht",
        "num_ell_to_school",
        "num_frl_to_school",
        "school_idx",
        "total_enrollment",
    ]
    for k in keys_to_keep:
        optout_estimates[k] = []

    group_keys = ["total", "black", "hispanic", "asian", "native", "white"]
    for i in range(0, len(df_combined)):
        for k in keys_to_keep:
            optout_estimates[k].append(df_combined[k][i])
        for g in group_keys:
            frac_to_keep = 1
            if df_combined["school_changed"][i]:
                frac_to_keep = (
                    1
                    - dist_choice["ratio_c_or_m_to_dist_{}".format(g)]
                    * choice_multiplier
                )

            # Take the floor to be conservative
            optout_estimates["num_{}_to_school".format(g)].append(
                np.floor(df_combined["num_{}_to_school".format(g)][i] * frac_to_keep)
            )
    df_soln_optouts = pd.DataFrame(data=optout_estimates)
    df_soln_optouts_g = group_by_school(df_soln_optouts, "new_school_nces")

    seg = compute_seg_measure(df_soln_optouts_g, key, ())["total"]
    norm_exp = compute_normalized_exposure_measure(df_soln_optouts_g, key, ())["total"]
    gini = compute_gini_measure(df_soln_optouts_g, key, ())["total"]

    return seg, norm_exp, gini


def compute_num_students_rezoned_measure(key, df_orig, df_soln, config):

    if config == "status_quo_zoning":
        return float("nan"), float("nan"), float("nan")

    df_orig = df_orig.rename(columns={"ncessch": "orig_nces_id"})
    df_soln_subset = df_soln[["block_id", "new_school_nces"]]
    df_combined = pd.merge(df_orig, df_soln_subset, on="block_id", how="inner")
    df_combined["school_changed"] = (
        df_combined["orig_nces_id"] != df_combined["new_school_nces"]
    )

    num_rezoned = np.sum(df_combined[key] * df_combined["school_changed"])
    total_num = df_combined[key].sum()
    percent_rezoned = num_rezoned / total_num
    return num_rezoned, percent_rezoned, total_num


def compute_travel_time_measure(df, key, val, extra=None):
    vals = []

    df, travel_time_matrix = extra
    for i in range(0, len(df)):
        try:
            travel_time = float(
                travel_time_matrix[df["block_idx"][i]][df["school_idx"][i]]
            )
        except Exception as e:
            # If for some reason we can't retrieve the travel time from the matrix
            travel_time = float("nan")
        num_from_cat_to_school = int(df[key][i])
        vals.extend([travel_time for j in range(0, num_from_cat_to_school)])

    return {
        "total": np.nansum(vals),
        "gini": gini(vals),
        "median": np.nanmedian(vals),
        "mean": np.nanmean(vals),
        "all": [],  # storing empty matrix for now since we don't need this in our analysis, can replace later with vals
    }


def compute_travel_time_for_rezoned_measure(df, travel_time_matrix, key):
    orig_travel_times = []
    rezoned_travel_times = []
    travel_time_diffs = []
    travel_time_percentage_diffs = []

    for i in range(0, len(df)):
        try:
            orig_travel_time = float(
                travel_time_matrix[df["block_idx"][i]][df["orig_school_idx"][i]]
            )
        except Exception as e:
            # If for some reason we can't retrieve the travel time from the matrix
            orig_travel_time = float("nan")
        try:
            rezoned_travel_time = float(
                travel_time_matrix[df["block_idx"][i]][df["school_idx"][i]]
            )
        except Exception as e:
            # If for some reason we can't retrieve the travel time from the matrix
            rezoned_travel_time = float("nan")

        num_from_cat_to_school = int(df[key][i])
        orig_travel_times.extend(
            [orig_travel_time for j in range(0, num_from_cat_to_school)]
        )
        rezoned_travel_times.extend(
            [rezoned_travel_time for j in range(0, num_from_cat_to_school)]
        )

        travel_time_diffs.extend(
            [
                rezoned_travel_time - orig_travel_time
                for j in range(0, num_from_cat_to_school)
            ]
        )
        travel_time_percentage_diffs.extend(
            [
                (rezoned_travel_time - orig_travel_time) / orig_travel_time
                for j in range(0, num_from_cat_to_school)
            ]
        )

    total_orig = np.nansum(orig_travel_times)
    gini_orig = gini(orig_travel_times)
    median_orig = np.nanmedian(orig_travel_times)
    mean_orig = np.nanmean(orig_travel_times)
    total_rezoned = np.nansum(rezoned_travel_times)
    gini_rezoned = gini(rezoned_travel_times)
    median_rezoned = np.nanmedian(rezoned_travel_times)
    mean_rezoned = np.nanmean(rezoned_travel_times)

    return {
        "total": total_orig,
        "gini": gini_orig,
        "median": median_orig,
        "mean": mean_orig,
        # "all": travel_time_diffs,
        "all": [],
        "total_diff": total_rezoned - total_orig,
        "gini_diff": gini_rezoned - gini_orig,
        "median_diff": median_rezoned - median_orig,
        "mean_diff": mean_rezoned - mean_orig,
        "total_change": (total_rezoned - total_orig) / total_orig,
        "gini_change": (gini_rezoned - gini_orig) / gini_orig,
        "median_change": (median_rezoned - median_orig) / median_orig,
        "mean_change": (mean_rezoned - mean_orig) / mean_orig,
    }


"""
	Computes index of dissimilarity, our core segregation measure
"""


def compute_seg_measure(df, key, val, extra=None):
    vals = []
    cat_total = df[key].sum()
    non_cat_total = df["num_total_to_school"].sum() - df[key].sum()
    for i in range(0, len(df)):
        vals.append(
            np.abs(
                (df[key][i] / cat_total)
                - ((df["num_total_to_school"][i] - df[key][i]) / non_cat_total)
            )
        )

    return {
        "total": 0.5 * np.nansum(vals),
        "gini": gini(vals),
        "median": np.nanmedian(vals),
        "mean": np.nanmean(vals),
        "all": [],  # storing empty matrix for now since we don't need this in our analysis, can replace later with vals
    }


"""
	Computes normalized exposure index
"""


def compute_normalized_exposure_measure(df, key, val, extra=None):
    vals = []
    for i in range(0, len(df)):
        vals.append(
            (df[key][i] / df[key].sum()) * (df[key][i] / df["num_total_to_school"][i])
        )

    P = df[key].sum() / df["num_total_to_school"].sum()

    return {
        "total": ((np.nansum(vals) - P) / (1 - P)),
        "gini": gini(vals),
        "median": np.nanmedian(vals),
        "mean": np.nanmean(vals),
        "all": [],  # storing empty matrix for now since we don't need this in our analysis, can replace later with vals
    }


"""
	Computes gini index
"""


def compute_gini_measure(df, key, val, extra=None):
    vals = []
    for i in range(0, len(df)):
        for j in range(0, len(df)):
            vals.append(
                (df["num_total_to_school"][i] * df["num_total_to_school"][j])
                * np.abs(
                    (df[key][i] / df["num_total_to_school"][i])
                    - (df[key][j] / df["num_total_to_school"][j])
                )
            )

    P = df[key].sum() / df["num_total_to_school"].sum()
    denom = 2 * (df["num_total_to_school"].sum() ** 2) * P * (1 - P)
    return {
        "total": np.nansum(vals) / denom,
        "gini": gini(vals),
        "median": np.nanmedian(vals),
        "mean": np.nanmean(vals),
        "all": [],  # storing empty matrix for now since we don't need this in our analysis, can replace later with vals
    }


def group_by_school(df, nces_key):
    df_g = (
        df.groupby(nces_key)
        .agg(
            {
                "num_white_to_school": "sum",
                "num_black_to_school": "sum",
                "num_asian_to_school": "sum",
                "num_native_to_school": "sum",
                "num_hispanic_to_school": "sum",
                "num_ell_to_school": "sum",
                "num_frl_to_school": "sum",
                "num_total_to_school": "sum",
                "district_perwht": "first",
                "district_perblk": "first",
                "district_pernam": "first",
                "district_perasn": "first",
                "district_perhsp": "first",
                "district_perell": "first",
                "district_perfrl": "first",
                "total_enrollment": "first",
                "school_idx": "first",
            }
        )
        .reset_index()
        .sort_values(by=nces_key)
    )
    return df_g


def handle_empty_dfs(cats, outcome_measures, outcome_vals, output_data):

    for c in cats:
        output_data["{}_num_rezoned".format(c)].append(float("nan"))
        output_data["{}_percent_rezoned".format(c)].append(float("nan"))
        output_data["{}_num_total".format(c)].append(float("nan"))
        for m in outcome_measures:
            output_data["{}_{}_travel_time_for_rezoned".format(c, m)].append(
                float("nan")
            )
            if not m == "all":
                output_data["{}_{}_travel_time_for_rezoned_change".format(c, m)].append(
                    float("nan")
                )
                output_data["{}_{}_travel_time_for_rezoned_diff".format(c, m)].append(
                    float("nan")
                )
            for v in outcome_vals:
                key = "{}_{}_{}".format(c, m, v)
                output_data[key].append(float("nan"))


def compute_metrics_for_zoning(
    cats,
    outcome_measures,
    outcome_vals,
    choice_multipliers,
    output_data,
    df_orig,
    df_orig_g,
    df_soln,
    df_soln_g,
    travel_time_matrix,
    config,
):

    output_data["config"].append(config)

    curr_df = df_soln
    curr_df_g = df_soln_g
    if config == "status_quo_zoning":
        curr_df = df_orig
        curr_df_g = df_orig_g

    # If the data frame is empty or non existent, store nans for all the keys
    if curr_df.empty or curr_df_g.empty:
        handle_empty_dfs(cats, outcome_measures, outcome_vals, output_data)
        return

    # Get df of only rezoned blocks
    curr_df_copy = curr_df.copy(deep=True)
    df_orig_copy = df_orig.copy(deep=True)[["block_idx", "school_idx"]].rename(
        columns={"school_idx": "orig_school_idx"}
    )
    merged_df = pd.merge(curr_df_copy, df_orig_copy, on="block_idx", how="inner")
    curr_df_rezoned_blocks = merged_df[
        merged_df["school_idx"] != merged_df["orig_school_idx"]
    ].reset_index(drop=True)

    outcome_val_functions = {
        "travel_time": (compute_travel_time_measure, (curr_df, travel_time_matrix)),
        "segregation": (compute_seg_measure, ()),
        "normalized_exposure": (compute_normalized_exposure_measure, ()),
        "gini": (compute_gini_measure, ()),
    }
    for c in cats:
        for v in outcome_vals:

            # Store the number
            val_results = outcome_val_functions[v][0](
                curr_df_g,
                CATS[c][0],
                CATS[c][1],
                extra=outcome_val_functions[v][1],
            )
            for m in outcome_measures:
                key = "{}_{}_{}".format(c, m, v)
                output_data[key].append(val_results[m])

        # Store the number rezoned per race per config
        (
            num_cat_rezoned,
            percent_cat_rezoned,
            num_cat_total,
        ) = compute_num_students_rezoned_measure(CATS[c][0], df_orig, df_soln, config)
        output_data["{}_num_rezoned".format(c)].append(num_cat_rezoned)
        output_data["{}_percent_rezoned".format(c)].append(percent_cat_rezoned)
        output_data["{}_num_total".format(c)].append(num_cat_total)

        # Store travel info specifically for those who were rezoned
        travel_info_for_rezoned = compute_travel_time_for_rezoned_measure(
            curr_df_rezoned_blocks, travel_time_matrix, CATS[c][0]
        )
        for m in travel_info_for_rezoned:
            m_split = m.split("_")
            m_0 = m_split[0]
            m_1 = ""
            if len(m_split) == 2:
                m_1 = "_" + m_split[1]

            key = "{}_{}_travel_time_for_rezoned{}".format(c, m_0, m_1)
            output_data[key].append(travel_info_for_rezoned[m])

        for s in choice_multipliers:
            seg, norm_exp, gini = estimate_outcomes_for_choice_scenario(
                CATS[c][0], df_orig, df_soln, s, config
            )
            output_data["{}_total_segregation_choice_{}".format(c, s)].append(seg)
            output_data["{}_total_normalized_exposure_choice_{}".format(c, s)].append(
                norm_exp
            )
            output_data["{}_total_gini_choice_{}".format(c, s)].append(gini)


def process_district_results(
    year,
    state,
    district_id,
    pre_solver_file,
    travel_time_matrix_file,
    solution_dir,
    output_dir,
):

    print(district_id)
    cats = ["black", "hisp", "ell", "frl", "white", "asian", "native"]
    outcome_measures = ["total", "mean", "median", "gini", "all"]

    # "segregation" is the dissimilarity index ... not changing it even after
    # adding these others for backward compat reasons ...
    outcome_vals = ["travel_time", "segregation", "normalized_exposure", "gini"]
    choice_scenarios = [0.5, 1]
    seg_vals_choice_scenarios = []
    for o in outcome_vals:
        if o == "travel_time":
            continue
        for c in choice_scenarios:
            seg_vals_choice_scenarios.append("{}_choice_{}".format(o, c))

    outcome_keys = []
    for c in cats:
        outcome_keys.append("{}_num_rezoned".format(c))
        outcome_keys.append("{}_percent_rezoned".format(c))
        outcome_keys.append("{}_num_total".format(c))
        for m in outcome_measures:
            for v in outcome_vals:
                outcome_keys.append("{}_{}_{}".format(c, m, v))
            for v in seg_vals_choice_scenarios:
                outcome_keys.append("{}_total_{}".format(c, v))
            outcome_keys.append("{}_{}_travel_time_for_rezoned".format(c, m))
            if not m == "all":
                outcome_keys.append("{}_{}_travel_time_for_rezoned_change".format(c, m))
                outcome_keys.append("{}_{}_travel_time_for_rezoned_diff".format(c, m))

    solver_metadata_keys = [
        "travel_time_threshold",
        "school_size_threshold",
        "objective_function",
        "cat_optimizing_for",
        "community_cohesion_threshold",
        "is_contiguous",
        "solver_status",
        "solver_runtime",
    ]
    output_data = {
        "config": [],
        "year": [year],
        "state": [state],
        "district_id": [district_id],
        "num_schools_in_district": [],
        "total_enrollment_in_district": [],
    }
    output_data.update([(k, []) for k in outcome_keys])
    output_data.update([(k, [float("nan")]) for k in solver_metadata_keys])
    df_orig = pd.read_csv(pre_solver_file.format(year, state, district_id))
    travel_time_matrix = read_dict(
        travel_time_matrix_file.format(year, state, district_id)
    )
    df_orig_g = group_by_school(df_orig, "ncessch")

    # Get num schools and total enrollment in district
    num_schools_in_district = len(df_orig_g)
    total_enrollment_in_district = df_orig_g["total_enrollment"].sum()

    # Compute metrics for status quo zoning
    output_data["num_schools_in_district"].append(num_schools_in_district)
    output_data["total_enrollment_in_district"].append(total_enrollment_in_district)
    compute_metrics_for_zoning(
        cats,
        outcome_measures,
        outcome_vals,
        choice_scenarios,
        output_data,
        df_orig,
        df_orig_g,
        None,
        None,
        travel_time_matrix,
        "status_quo_zoning",
    )

    # Iterate over rezoning configs to compute metrics
    for config in os.listdir(solution_dir + district_id + "/"):
        # print(config)
        config_params = config.split("_")
        travel_time_threshold = config_params[0]
        school_size_threshold = config_params[1]
        cat = config_params[2]
        objective_function = "_".join(config_params[3:5])

        # If for some reason we have a relic of a previous job before we added the community constraints
        # skip it
        try:
            community_cohesion_threshold = config_params[6]
        except Exception as e:
            continue

        # If for some reason we have a relic of a previous job before we added the contiguity constraints
        # skip it
        try:
            is_contiguous = config_params[7]
        except Exception as e:
            continue

        solution_file = ""
        other_file = ""
        for f in os.listdir(solution_dir + district_id + "/" + config + "/"):
            if f.startswith("solution"):
                solution_file = f
            if f.startswith("solver"):
                other_file = f

        df_soln = pd.DataFrame()
        df_soln_g = pd.DataFrame()
        if solution_file:
            df_soln = pd.read_csv(
                solution_dir + district_id + "/" + config + "/" + solution_file
            )
            df_soln = add_idx_fields(df_orig, df_soln)
            df_soln_g = group_by_school(df_soln, "new_school_nces")
            solution_params = solution_file.split("_")
            solver_runtime = solution_params[2]
            solver_status = get_name_for_solver_status(solution_params[3])
        elif other_file:
            solver_runtime = float("nan")
            solver_status = get_name_for_solver_status(
                read_dict(solution_dir + district_id + "/" + config + "/" + other_file)[
                    "status"
                ]
            )
        else:
            solver_runtime = float("nan")
            solver_status = float("nan")

        # Store non-cat-dependent data
        output_data["year"].append(year)
        output_data["state"].append(state)
        output_data["district_id"].append(district_id)
        output_data["travel_time_threshold"].append(travel_time_threshold)
        output_data["school_size_threshold"].append(school_size_threshold)
        output_data["community_cohesion_threshold"].append(community_cohesion_threshold)
        output_data["is_contiguous"].append(is_contiguous)
        output_data["cat_optimizing_for"].append(cat)
        output_data["objective_function"].append(objective_function)
        output_data["solver_status"].append(solver_status)
        output_data["solver_runtime"].append(solver_runtime)
        output_data["num_schools_in_district"].append(num_schools_in_district)
        output_data["total_enrollment_in_district"].append(total_enrollment_in_district)

        # Store cat-dependent data
        compute_metrics_for_zoning(
            cats,
            outcome_measures,
            outcome_vals,
            choice_scenarios,
            output_data,
            df_orig,
            df_orig_g,
            df_soln,
            df_soln_g,
            travel_time_matrix,
            config,
        )

    df_out = pd.DataFrame(data=output_data)
    output_dir = output_dir + solution_dir
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_dir + "analysis_{}.csv".format(district_id), index=False)
    print("Done with {}".format(district_id))


def output_csv_for_exploratory_analyses(
    solution_dir="models/results/{}",
    pre_solver_file="models/solver_files/{}/{}/prepped_file_for_solver_{}.csv",
    travel_time_matrix_file="models/solver_files/{}/{}/prepped_travel_time_matrix_{}.csv",
    year="2122",
    output_dir="data/prepped_csvs_for_analysis/",
    processing_function=process_district_results,
):

    N_THREADS = 5
    solution_dir = solution_dir.format(year)
    all_districts_to_process = []
    for state in os.listdir(solution_dir):
        curr_solution_dir = os.path.join(solution_dir, state, "")
        for district_id in os.listdir(curr_solution_dir):
            all_districts_to_process.append(
                (
                    year,
                    state,
                    district_id,
                    pre_solver_file,
                    travel_time_matrix_file,
                    curr_solution_dir,
                    output_dir,
                )
            )

    # all_districts_to_process = [
    #     (
    #         year,
    #         "TN",
    #         "4703180",
    #         pre_solver_file,
    #         travel_time_matrix_file,
    #         solution_dir.format(year) + "/TN" + "/",
    #         output_dir,
    #     )
    # ]
    # for a, b, c, d, e, f, g in all_districts_to_process:
    #     processing_function(a, b, c, d, e, f, g)
    #     exit()

    print("Starting parallel processing ...")
    from multiprocessing import Pool

    p = Pool(N_THREADS)
    p.starmap(processing_function, all_districts_to_process)

    p.terminate()
    p.join()


if __name__ == "__main__":
    # district_id = "5102670"
    # year = "2122"
    # state = "VA"
    # solution_dir = "simulation_outputs/va_2122_exposure/2122/VA/{}/".format(district_id)
    # cat_optimizing_for = "black"
    # school_nces = "510267001101"
    # analyze_school_distributions_pre_post(
    #     district_id, year, state, solution_dir, cat_optimizing_for, school_nces
    # )
    output_csv_for_exploratory_analyses()
    # add_contiguous_suffix_to_local_solution_dirs()
    # remove_solution_dirs()
    # filter_and_update_consolidated_sims_file()
