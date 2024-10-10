from collections import defaultdict
import glob
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

import config as app_config
import app_utils
from config import DATA_DIR
import select_boxes


@st.cache_data
def get_global_data():
    # Points to CSVs with all the simulation outputs we care about
    sim_results = [
        DATA_DIR
        + "simulation_outputs/2122_top_100_dissim_longer/consolidated_simulation_results.csv",
        DATA_DIR
        + "simulation_outputs/2122_all_usa_dissim/consolidated_simulation_results.csv",
        DATA_DIR
        + "simulation_outputs/2122_shaker_heights_expanded_dissim/consolidated_simulation_results.csv",
        ## NG: old ones below
        # DATA_DIR
        # + "simulation_outputs/va_2122_exposure/consolidated_simulation_results_filtered.csv",
        # DATA_DIR
        # + "simulation_outputs/va_2122_contiguous/consolidated_simulation_results.csv",
        # DATA_DIR
        # + "simulation_outputs/all_usa_2122/consolidated_simulation_results.csv",
        # DATA_DIR
        # + "simulation_outputs/nc_wake_2122/consolidated_simulation_results.csv",
    ]
    df_sim_results = pd.concat([pd.read_csv(s) for s in sim_results]).astype(
        {"district_id": "str"}
    )
    df_sim_results["district_id"] = app_utils.add_leading_zero_for_dist(
        df_sim_results, "district_id"
    )

    filename_state_codes = DATA_DIR + "data/state_codes.csv"
    df_states = pd.read_csv(filename_state_codes, dtype="str")
    # df_sim_results is the data frame with all info. info organized by district.
    return df_states, df_sim_results


@st.cache_data
def get_state_data(state):
    filename_school_state_data = (
        DATA_DIR + "data/derived_data/2122/%s/schools_file_for_assignment.csv" % (state)
    )
    df_school_state_data = pd.read_csv(filename_school_state_data)
    df_school_state_data = df_school_state_data.astype(
        {"ncessch": "str", "leaid": "str"}
    )
    df_school_state_data["leaid"] = app_utils.add_leading_zero_for_dist(
        df_school_state_data, "leaid"
    )
    df_school_state_data["ncessch"] = app_utils.add_leading_zero_for_school(
        df_school_state_data, "ncessch"
    )
    df_school_state_data.insert(0, "Name", df_school_state_data.pop("SCH_NAME"))
    return df_school_state_data


@st.cache_data
def get_district_data(selected_state, selected_district, df_school_state_data):
    filename_district_shape = (
        DATA_DIR + "data/census_block_shapefiles_2020/{}-{}/{}-{}-{}.geodata.csv"
    )
    year = "2122"

    # Get district shape info for selected district, and preprocess it a little
    df_district_shapes = gpd.GeoDataFrame(
        pd.read_csv(
            filename_district_shape.format(
                year,
                selected_state,
                year,
                selected_state,
                selected_district,
            )
        ).astype({"ncessch": "str"})
    )
    df_district_shapes["ncessch"] = app_utils.add_leading_zero_for_school(
        df_district_shapes, "ncessch"
    )
    df_district_shapes["geometry"] = gpd.GeoSeries.from_wkt(
        df_district_shapes["geometry"]
    ).simplify(0.0002)

    df_district_shapes = df_district_shapes.set_geometry("geometry")
    df_district_shapes.crs = "epsg:4326"
    matching_school_info = df_school_state_data[
        df_school_state_data["leaid"] == selected_district
    ]
    return df_district_shapes, matching_school_info


def get_solution_info(
    selected_state,
    selected_district,
    df_sim_results,
    config_code,
):
    # Solutions files (glob that matches them)
    glob_solutions_files = DATA_DIR + "/results/2122*dissim*/2122/{}/{}/{}"
    # glob_solutions_files = DATA_DIR + "/simulation_outputs/2122*dissim*/2122/{}/{}/{}"

    config_results = df_sim_results.loc[
        (df_sim_results["config"] == config_code)
        & (df_sim_results["district_id"] == str(selected_district))
    ]

    if config_results.empty:
        return None
    else:
        df_solution = pd.read_csv(
            find_file_by_suffix(
                glob_solutions_files.format(
                    selected_state, selected_district, config_code
                ),
                ".csv",
            )
        ).astype({"ncessch_x": "str", "ncessch_y": "str"})
        df_solution["ncessch_x"] = app_utils.add_leading_zero_for_school(
            df_solution, "ncessch_x"
        )
        df_solution["ncessch_y"] = app_utils.add_leading_zero_for_school(
            df_solution, "ncessch_y"
        )
        return df_solution, config_results


def get_summary_school_changes(df_solution, matching_school_info):
    from_df = df_solution.groupby(["ncessch_x"]).sum()
    to_df = df_solution.groupby(["ncessch_y"]).sum()
    joined_df = from_df.join(to_df, rsuffix="_new")
    joined_df = joined_df.join(
        matching_school_info.set_index("ncessch"), rsuffix="_info"
    )
    joined_df["Name"] = joined_df["Name"].apply(
        lambda x: select_boxes.pretty_print_school_name(str(x))
    )
    # Each row is a school, each column is a demographic group.  In each cell
    # is a before-after for the count of students
    joined_df = joined_df.sort_values(by=["num_total_to_school_new"], ascending=False)

    # Compute fraction of school comprised of each student group
    # Then, identify and store schools that have an
    # over-concentration of students relative to the district
    data_to_return = defaultdict(dict)
    all_cats = list(app_config.CAT_MAPPING.keys())
    all_cats.append("non_white")
    for cat in all_cats:

        if cat == "non_white":

            joined_df["school_non_white"] = (
                joined_df["num_total_to_school"] - joined_df["num_white_to_school"]
            ) / joined_df["num_total_to_school"]

            joined_df["school_non_white_new".format(cat)] = (
                joined_df["num_total_to_school_new"]
                - joined_df["num_white_to_school_new"]
            ) / joined_df["num_total_to_school_new"]

        else:

            joined_df["school_{}".format(cat)] = (
                joined_df["num_{}_to_school".format(cat)]
                / joined_df["num_total_to_school"]
            )
            joined_df["school_{}_new".format(cat)] = (
                joined_df["num_{}_to_school_new".format(cat)]
                / joined_df["num_total_to_school_new"]
            )

        over_conc_school_props = []
        updated_school_props = []
        school_names = []
        dist_per = ""
        for _, school in joined_df.iterrows():

            if cat == "non_white":
                if school["school_{}".format(cat)] > 1 - school["district_perwht_info"]:
                    dist_per = 1 - school["district_perwht_info"]
                    over_conc_school_props.append(school["school_{}".format(cat)])
                    updated_school_props.append(school["school_{}_new".format(cat)])
                    school_names.append(school["Name"])
            else:
                if (
                    school["school_{}".format(cat)]
                    > school["district_{}_info".format(app_config.CAT_MAPPING[cat])]
                ):
                    dist_per = school[
                        "district_{}_info".format(app_config.CAT_MAPPING[cat])
                    ]
                    over_conc_school_props.append(school["school_{}".format(cat)])
                    updated_school_props.append(school["school_{}_new".format(cat)])
                    school_names.append(school["Name"])

        group_code = cat
        if cat == "hispanic":
            group_code = "hisp"

        data_to_return[group_code]["district_frac"] = 100 * dist_per
        data_to_return[group_code]["num_schools_over_conc"] = len(
            over_conc_school_props
        )
        data_to_return[group_code]["num_total_schools"] = len(joined_df)
        data_to_return[group_code]["avg_overconc_frac_pre"] = 100 * np.nanmean(
            over_conc_school_props
        )
        data_to_return[group_code]["avg_overconc_frac_post"] = 100 * np.nanmean(
            updated_school_props
        )
        data_to_return[group_code]["pre_post_pairs"] = list(
            zip(over_conc_school_props, updated_school_props)
        )
        data_to_return[group_code]["school_names"] = school_names
        diff_pre_post = np.array(over_conc_school_props) - np.array(
            updated_school_props
        )

        if len(diff_pre_post) > 0:
            data_to_return[group_code]["ind_for_greatest_change"] = np.argmax(
                diff_pre_post
            )
        else:
            data_to_return[group_code]["ind_for_greatest_change"] = float("nan")

    return data_to_return


@st.cache_data
def get_change_measures(
    status_quo_results, config_results, selected_district, groups, change_types
):
    """
    Displays the changes in average travel time and segregation level for each socioeconomic group
    """

    all_findings = {}  # change_type -> group_findings

    for change_type in change_types:
        group_findings = {}

        new_val_weighted_sum = 0.0
        new_val_count = 0

        for group in groups:
            new_val_for_group = config_results.iloc[0][group + "_" + change_type]
            old_val_for_group = status_quo_results.iloc[0][group + "_" + change_type]
            if change_type.startswith("mean_travel_time"):
                # Convert travel times from seconds to minutes
                new_val_for_group /= 60.0
                old_val_for_group /= 60.0
            elif change_type == "percent_rezoned":
                # These are actually out of 1 in the raw data; display as percentages
                new_val_for_group *= 100.0
                old_val_for_group *= 100.0
            delta = float(new_val_for_group - old_val_for_group)
            if old_val_for_group == 0.0:
                change_as_increase_rate = 0.0
            else:
                change_as_increase_rate = delta / float(old_val_for_group)

            group_findings[group] = (
                delta,
                old_val_for_group,
                new_val_for_group,
            )

            # If we're probably gonna get a nan for a given group, skip it
            if (
                np.isnan(new_val_for_group)
                or config_results.iloc[0][group + "_num_total"] == 0
            ):
                continue

            new_val_weighted_sum += (
                new_val_for_group * config_results.iloc[0][group + "_num_total"]
            )
            new_val_count += config_results.iloc[0][group + "_num_total"]

        try:
            group_findings["weighted_avg"] = new_val_weighted_sum / new_val_count
        except Exception as e:
            group_findings["weighted_avg"] = 0
        all_findings[change_type] = group_findings

    return all_findings


def find_file_by_suffix(directory, suffix):
    # Find it on local filesystem
    filenames = glob.glob(os.path.join(directory, "*" + suffix))
    if filenames:
        return filenames[0]
    return None
