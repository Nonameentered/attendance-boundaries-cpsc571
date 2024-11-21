"""
Displays the select boxes that let viewers pick the inputs and optimization constraints
"""

from collections import defaultdict
import streamlit as st
import config as app_config


def read_static(fname):
    return open("static/" + fname + ".txt").read()


def states(df_states, df_sim_results, default):
    """
    Displays select box for user to display a state and returns the state that the user selects.
    """
    states_with_data = list(df_sim_results["state"].unique())
    selected_state = st.selectbox(
        "State",
        states_with_data,
        format_func=lambda x: df_states.loc[df_states["abbrev"] == x].iloc[0]["name"],
        index=states_with_data.index(default),
    )
    return selected_state


def districts(df_school_state_data, df_sim_results, default):

    districts = list(df_school_state_data["leaid"].unique())
    simulated_districts = list(df_sim_results["district_id"].unique())
    schools_grouped_by_district = df_school_state_data.groupby(["leaid"])

    # Only show districts with > 1 school, and that have simulation results
    districts = list(
        filter(
            lambda x: (
                x in simulated_districts
                and len(schools_grouped_by_district.indices[x]) > 1
            ),
            districts,
        )
    )

    districts.sort()  # IDs correspond to alpha sorting already

    # District selectbox
    selected_district = st.selectbox(
        "School district",
        districts,
        format_func=lambda x: "%s (%d elementary schools)"
        % (
            pretty_print_district_name(
                df_school_state_data.set_index("leaid").xs(str(x))["LEA_NAME"][0]
            ),
            len(schools_grouped_by_district.indices[x]),
        ),
        index=(default and default in districts and districts.index(default)) or 0,
    )
    return selected_district


def schools(df_school_state_data):
    all_schools = df_school_state_data["Name"]
    selected_school = st.selectbox(
        "School", all_schools, format_func=lambda x: pretty_print_school_name(x)
    )
    return selected_school


def travel_configs(options=[0.5, 1], default=1):
    options_display_map = {
        x: ((x == 0.0) and "No increase allowed" or "%d%% increase" % (x * 100))
        for x in options
    }
    travel_time_increase = st.selectbox(
        "Maximum travel time increase",
        [options_display_map[x] for x in options],
        help=read_static("tooltip_traveltime"),
        index=default,
    )
    return str(
        [
            key
            for key, value in options_display_map.items()
            if value == travel_time_increase
        ][0]
    )


def optimization_targets(default):
    target = st.selectbox(
        "Category of students for which to maximize diversity",
        app_config.OPTIMIZATION_TARGETS.values(),
        format_func=lambda x: app_config.OPTIMIZATION_TARGETS_DISPLAY[x].capitalize(),
        index=list(app_config.OPTIMIZATION_TARGETS.values()).index(default),
        help=read_static("tooltip_category"),
    )
    return target


def size_increases(options=[0.1, 0.15, 0, 2], default=1):
    options_display_map = {
        x: ((x == 0.0) and "No increase allowed" or "%d%% increase" % (x * 100))
        for x in options
    }
    school_size_increase = st.selectbox(
        "Maximum school size increase",
        [options_display_map[x] for x in options],
        help=read_static("tooltip_schoolsize"),
        index=default,
    )
    return str(
        [
            key
            for key, value in options_display_map.items()
            if value == school_size_increase
        ][0]
    )


def contiguity(options=["True", "False"], default="True"):
    """
    Displays check box for contiguity.
    """
    st.text(
        ""
    )  # For vertical alignment (see https://discuss.streamlit.io/t/vertically-center-the-content-of-columns/6163)
    st.text("")
    contiguous = st.checkbox(
        "Maintain contiguity",
        value=(default == "True"),
        help=read_static("tooltip_contiguous"),
    )
    return contiguous


def render_geo_inputs(query_params, df_states, df_sim_results, get_state_data):
    """Render selectboxes for the state and district."""
    state_input_region, district_input_region = st.columns([1, 3])
    with state_input_region:
        default_state = query_params["state"][0] if "state" in query_params else "GA"
        selected_state = states(df_states, df_sim_results, default_state)
    df_school_state_data = get_state_data(selected_state)
    with district_input_region:
        default_district = (
            query_params["district"][0] if "district" in query_params else "1302220"
        )
        selected_district = districts(
            df_school_state_data, df_sim_results, default_district
        )
        selected_district_name = pretty_print_district_name(
            df_school_state_data.set_index("leaid").xs(selected_district)["LEA_NAME"][0]
        )
    return (
        selected_state,
        selected_district,
        selected_district_name,
        df_school_state_data,
    )


def render_simulation_inputs(df_sim_results, query_params=None):
    """
    The default for the selectboxes is also the default simulation that users see
    when they visit the dashboard, and the logic to determine the default is somewhat
    hairy.  Any parameters in the URL take precedence, and only configurations that meet
    the constraints in config.config_code_qualifies_for_default should be considered.
    Given those constraints, we want the default settings to reflect the best
    configuration (the one that maximizes app_config.OBJECTIVE_COLUMN).
    """

    # First off, limit the sim results to the ones that are eligible for default
    df_sim_results_qualified = df_sim_results[
        df_sim_results["config"].apply(
            lambda code: app_config.config_code_qualifies_for_default(code)
        )
    ]

    # If a target group is specified in the URL parameters, we'll find
    # the best config in terms of how config.OBJECTIVE_COLUMN impacts that
    # group;  otherwise, consider all groups.   (Note that "target_group" has nothing to do
    # with the group for which the optimization was run and that appears in the config code;
    # we find the best config in terms of the actual results for that group, which could
    # be in a config for which it wasn't targeted.)
    if query_params and "target_group" in query_params:
        target_groups_to_scan = [query_params["target_group"][0]]
    else:
        target_groups_to_scan = list(app_config.OPTIMIZATION_TARGETS.values())

    # Find the configuration that maximizes the objection function for any of
    # target_groups_to_scan, subject to the qualifications defined in app_config,
    min_value_overall = 2.0
    for group in target_groups_to_scan:
        min_val_for_group = df_sim_results_qualified[
            group + app_config.OBJECTIVE_COLUMN
        ].max()
        if min_val_for_group <= min_value_overall:
            min_value_overall = min_val_for_group
            default_target_group = group

    # Now render the target group selectbox.  The final objective to use is determined
    # by the stqte of this selectbox.
    target_group_code = optimization_targets(default_target_group)
    objective = target_group_code + app_config.OBJECTIVE_COLUMN

    # Now find the best config code that matches the user's selected group code
    best_code = df_sim_results_qualified[
        (
            df_sim_results_qualified[objective]
            == df_sim_results_qualified[objective].max()
        )
    ]["config"].iloc[0]

    # We override the config code to use if it's specified in the params.
    if query_params and "config_code" in query_params:
        default_code = query_params["config_code"][0]
    else:
        default_code = best_code

    if default_code == app_config.STATUS_QUO_ZONING:
        # We can't do any better than the status quo, so let's break out early and tell the user that.
        # TODO:  Do something better here
        return None

    # Find all the unique values of the exposed config settings available as simulations
    values = defaultdict(lambda: set())
    for code in df_sim_results["config"]:
        if code != app_config.STATUS_QUO_ZONING:
            for k, v in app_config.parse_config_code(code).items():
                values[k].add(v)
    for k, v in values.items():
        values[k] = list(values[k])
        values[k].sort()

    default_code_parts = app_config.parse_config_code(default_code)
    columns = st.columns(3)
    with columns[0]:
        travel_config_code = travel_configs(
            options=values["travel"],
            default=values["travel"].index(default_code_parts["travel"]),
        )
    with columns[1]:
        size_config_code = size_increases(
            options=values["size"],
            default=values["size"].index(default_code_parts["size"]),
        )
    with columns[2]:
        contiguity_config_code = contiguity(
            options=["True", "False"],
            default="True",
        )

    # Now find the best config that matches the given constraints
    min_objective = 2.0
    ret = None
    for _, r in df_sim_results.iterrows():
        config = app_config.parse_config_code(r["config"])
        if config and (
            str(config["travel"]) == travel_config_code
            and str(config["size"]) == size_config_code
            and str(config["is_contiguous"]) == str(contiguity_config_code)
            and float(config["cohesion"])
            >= (0.5 if str(config["is_contiguous"]) == "True" else 0)
        ):
            if r[objective] < min_objective:
                ret = r["config"]
                min_objective = r[objective]
    return {"best": best_code, "selected": ret, "target_group": target_group_code}


def pretty_print_district_name(s):
    s = (
        s.replace(" CO ", " County ")
        .replace(" PBLC ", " Public ")
        .replace(" SCHS", " Schools")
    )
    return s.title()


def pretty_print_school_name(s):
    s = (" " + s.title() + " ").replace(" Elem ", " Elementary ").strip()
    return s
