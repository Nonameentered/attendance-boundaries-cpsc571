import streamlit as st

import config as app_config
import data
import logger
import maps
import impact_plots
import select_boxes
import landing_page

# import referral_codes


def init():
    st.set_page_config(page_title=app_config.TITLE, layout="wide")
    st.title(app_config.TITLE)

    st.session_state["access_obtained"] = True

    if not "access_obtained" in st.session_state:
        st.session_state["access_obtained"] = False
    if not "continue_pressed" in st.session_state:
        st.session_state["continue_pressed"] = False

    content_placeholders = []
    # if not st.session_state["access_obtained"]:
    #     content_placeholders = landing_page.check_consent()

    return content_placeholders


def include_markdown(fname, replace={}, highlight_numbers=False):
    text = open("static/" + fname + ".md").read()
    for k, v in replace.items():
        if highlight_numbers and v[0].isnumeric() or v[0].startswith("-"):
            v = '<mark style="background-color: #fdfd96;">' + v + "</mark>"
        text = text.replace("${%s}" % (k.upper()), v)
    st.markdown(text, unsafe_allow_html=True)


def app():
    logger.init()
    query_params = st.experimental_get_query_params()
    logger.log("START", (), query_params=query_params)
    content_placeholders = init()

    st.session_state["continue_pressed"] = True
    if ("in_study" in query_params and query_params["in_study"][0] == "0") or (
        st.session_state["continue_pressed"]
    ):
        # if True:
        st.session_state["access_obtained"] = True

        # Clear out all first page placeholders
        for p in content_placeholders:
            p.empty()

        # Remove hamburger at the top right
        st.markdown(
            """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;}/style> """,
            unsafe_allow_html=True,
        )

        df_states, df_sim_results = data.get_global_data()
        #        campaign_info = referral_codes.load_campaign_info()

        # if st.session_state["referral_code"] and False:
        #     user_info = campaign_info.get(st.session_state["referral_code"], {})
        #     # Set the state and district by overriding query params
        #     query_params["state"] = [user_info["source_data"]["state"]]
        #     query_params["district"] = [user_info["source_data"]["leaid"]]

        # Intro text
        include_markdown("welcome_message")
        # Dynamic text that describes the results of the best simulation for the target
        # Initially display a "blank state" solution summary text with placeholders
        geo_selectors_placeholder = st.container()
        st.header("Results")
        solution_summary_text = st.empty()
        # Placeholders for content which is generated later
        map_placeholder = st.container()
        survey_placeholder = st.empty()
        tweak_parameters_placeholder = st.container()
        more_info_placeholder = st.container()

        # State and district selectors
        with geo_selectors_placeholder:
            (
                selected_state,
                selected_district,
                selected_district_name,
                df_school_state_data,
            ) = select_boxes.render_geo_inputs(
                query_params, df_states, df_sim_results, data.get_state_data
            )

        df_district_shapes, matching_school_info = data.get_district_data(
            selected_state, selected_district, df_school_state_data
        )

        blank_state_replacements = {
            "DISTRICT": selected_district_name,
            "ATTEMPT_DESCRIPTION": "best attempted boundary redrawing",
            "CATEGORY": "...",
            "CAT_DISTRICT_PERCENT": "...",
            "NUM_OVER_CONC_SCHOOLS": "...",
            "NUM_TOTAL_SCHOOLS": "...",
            "AVG_CONC_STATUS_QUO": "...",
            "AVG_CONC_NEW": "...",
            "MAX_CHANGING_SCHOOL": "...",
            "MAX_CHANGING_PRE": "...",
            "MAX_CHANGING_POST": "...",
            "SCHOOL_SWITCH_PERCENT": "...",
            "TRAVEL_INCREASE_MINUTES": "...",
        }
        with solution_summary_text:
            include_markdown("results_summary", replace=blank_state_replacements)

        # Simulation configuration controls.  We get all the simulation results relevant to the
        # district and pass them into the function that generates the selectboxes, so that the
        # optimal combination is set as the default.
        df_sim_results = df_sim_results[
            df_sim_results["district_id"] == selected_district
        ]

        with tweak_parameters_placeholder:
            with st.expander(
                "Click here to change values and see another boundary scenario", True
            ):
                include_markdown(
                    "constraint_description",
                    replace={"NUM_CONFIGS": str(len(df_sim_results.index))},
                )
                config_codes = select_boxes.render_simulation_inputs(
                    df_sim_results, query_params
                )
                target_group_code = config_codes["target_group"]
                config_code = config_codes["selected"]

        if not config_code:
            st.header(
                "We don't seem to have results yet for this district.  Please email us at {} if you'd like to learn more.".format(
                    app_config.EMAIL
                )
            )
            with solution_summary_text:
                st.header("No results yet for this district.")
            return

        df_solution, config_results = data.get_solution_info(
            selected_state,
            selected_district,
            df_sim_results,
            config_code,
        )

        if not config_results.empty:
            # Analyze results
            status_quo_results = df_sim_results.loc[
                (df_sim_results["config"] == app_config.STATUS_QUO_ZONING)
                & (df_sim_results["district_id"] == str(selected_district))
            ]

            school_info = data.get_summary_school_changes(
                df_solution, matching_school_info
            )

            change_measures = data.get_change_measures(
                status_quo_results,
                config_results,
                selected_district,
                ["black", "white", "hisp", "asian", "native"],
                [
                    "num_total",
                    "mean_travel_time",
                    "mean_travel_time_for_rezoned_diff",
                    "mean_xexposure_prob",
                    "mean_xexposure_prob_change",
                    "percent_rezoned",
                    # "total_xexposure_num_diff",
                ],
            )

            with solution_summary_text:
                # Now update the solution summary text with real values

                # For white, we special case to show everything in terms of non-White concentration
                # since that's where we often see changes in districts (and schools that have a high)
                # concentration of non-white students
                curr_group_code = (
                    "non_white" if target_group_code == "white" else target_group_code
                )
                replacements = blank_state_replacements
                replacements.update(
                    {
                        "CATEGORY": (
                            app_config.OPTIMIZATION_TARGETS_DISPLAY[target_group_code]
                            if target_group_code != "white"
                            else "non-White students"
                        ),
                        "CAT_DISTRICT_PERCENT": "%.1f%%"
                        % (school_info[curr_group_code]["district_frac"]),
                        "NUM_OVER_CONC_SCHOOLS": "%s"
                        % (school_info[curr_group_code]["num_schools_over_conc"]),
                        "NUM_TOTAL_SCHOOLS": "%s"
                        % (school_info[curr_group_code]["num_total_schools"]),
                        "AVG_CONC_STATUS_QUO": "%.1f%%"
                        % (school_info[curr_group_code]["avg_overconc_frac_pre"]),
                        "AVG_CONC_NEW": "%.1f%%"
                        % (school_info[curr_group_code]["avg_overconc_frac_post"]),
                        "MAX_CHANGING_SCHOOL": "%s"
                        % school_info[curr_group_code]["school_names"][
                            school_info[curr_group_code]["ind_for_greatest_change"]
                        ],
                        "MAX_CHANGING_PRE": "%.1f%%"
                        % (
                            100
                            * school_info[curr_group_code]["pre_post_pairs"][
                                school_info[curr_group_code]["ind_for_greatest_change"]
                            ][0]
                        ),
                        "MAX_CHANGING_POST": "%.1f%%"
                        % (
                            100
                            * school_info[curr_group_code]["pre_post_pairs"][
                                school_info[curr_group_code]["ind_for_greatest_change"]
                            ][1]
                        ),
                        "SCHOOL_SWITCH_PERCENT": "%.1f%%"
                        % (change_measures["percent_rezoned"]["weighted_avg"]),
                        "TRAVEL_INCREASE_MINUTES": "%.1f"
                        % (
                            # change_measures["mean_travel_time_for_rezoned_diff"][
                            #     target_group_code
                            # ][2]
                            change_measures["mean_travel_time_for_rezoned_diff"][
                                "weighted_avg"
                            ]
                        ),
                    }
                )
                include_markdown(
                    "results_summary", replace=replacements, highlight_numbers=True
                )

            logger.log(
                "DASHBOARD_VIEW",
                (
                    selected_state,
                    selected_district,
                    target_group_code,
                    config_code,
                    config_codes["best"],
                ),
                query_params=query_params,
            )
            # Render teaser for Qualtrics survey
            # config = app_config.parse_config_code(config_code)
            # qualtrics_url = app_config.QUALTRICS_SURVEY_URL.format(
            #     **{
            #         "session_id": "0",
            #         "source_id": "email",
            #         "travel_time_threshold": str(config["travel"]),
            #         "school_size_threshold": str(config["size"]),
            #         "student_cat": target_group_code,
            #         "obj_fcn": "min_total",
            #         "referral_code": ""
            #         # if "referral_code" not in st.session_state
            #         # else st.session_state["referral_code"],
            #         "is_contiguous": str(config["is_contiguous"]),
            #         "set": "" if "set" not in query_params else query_params["set"][0],
            #     }
            # )
            # style = "<style> .big-font { font-size:30px !important ; padding: 5px; background-color: lightgreen; } </style>"
            # with survey_placeholder:
            #     # st.info(
            #     #     "What do you think about this change?  Take [this survey](%s)!"
            #     #     % (qualtrics_url)
            #     # )
            #     st.markdown(
            #         style
            #         + '<p class="big-font"><a href="%s" target="_blank">Tell us what you think</a> about these boundaries!</p>'
            #         % (qualtrics_url),
            #         unsafe_allow_html=True,
            #     )

            with more_info_placeholder:
                # Render additional graphs about the change
                # st.header("More information about these alternative boundaries")
                include_markdown("impact_description")
                school_figs = impact_plots.display_school_info(
                    df_solution,
                    matching_school_info,
                    "Demographic changes in individual schools",
                )
                impact_plots.display_change_measures(
                    change_measures,
                    "Percent of students rezoned",
                    "percent_rezoned",
                    show_current=False,
                )
                impact_plots.display_change_measures(
                    change_measures, "Travel time (minutes)", "mean_travel_time"
                )
                # impact_plots.display_change_measures(
                #     change_measures, "Chances for diverse exposures", "mean_xexposure_prob"
                # )

            # Render maps
            with map_placeholder:
                include_markdown("map_description")
                maps.write_maps(
                    matching_school_info,
                    df_district_shapes,
                    df_solution,
                    selected_district + "-" + config_code,
                    school_figs,
                )

            # Also put one at the end in case they missed it.
            st.markdown(
                style
                + '<p class="big-font">Remember to <a href="%s" target="_blank">tell us what you think</a> about these boundaries!</p>'
                % (qualtrics_url),
                unsafe_allow_html=True,
            )

            st.write(
                "Permanent link to this page:  **%s**"
                % (
                    app_config.get_permalink(
                        selected_state,
                        selected_district,
                        target_group_code,
                        config_code,
                        config_codes["best"],
                    )
                )
            )

            # st.components.v1.html(
            #     '<img width=0 height=0 src="/s2/a.png?referral_code='
            #     + str(st.session_state.get("referral_code"))
            #     + '">'
            # )


if __name__ == "__main__":
    app()
