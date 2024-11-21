"""
Methods that display the impact info panels (change measures, school breakdown) on the streamlit app
"""

import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as graph_objects
import plotly.io
from plotly.subplots import make_subplots

import config
import select_boxes

from collections import defaultdict
import textwrap


def display_change_measures(all_findings, title, key, show_current=True):
    group_findings = list(
        [x for x in all_findings[key].items() if x[0] != "weighted_avg"]
    )

    boundaries = []
    bar_titles = []
    if show_current:
        boundaries = ["current"] * len(group_findings)
        bar_titles = [k[1][1] for k in group_findings]
    boundaries.extend(["proposed"] * len(group_findings))
    bar_titles.extend([k[1][2] for k in group_findings])

    source = pd.DataFrame(
        {
            "Group": [config.OUTCOME_TARGETS_DISPLAY[k[0]] for k in group_findings]
            * (show_current and 2 or 1),
            "Boundaries": boundaries,
            title: bar_titles,
        }
    )
    # Use plotly for this.  (Altair's grouped bar charts seem broken on streamlit)
    with st.expander(title, expanded=False):
        chart = px.bar(
            source,
            x="Group",
            y=title,
            color="Boundaries",
            barmode="group",
            color_discrete_sequence=(
                show_current and config.COLORSCALE or config.COLORSCALE[1:]
            ),
        )
        if not show_current:
            chart.layout.update(showlegend=False)
        st.plotly_chart(chart)


def display_school_info(df_solution, matching_school_info, title):
    """Returns a map of ncessch -> plotly figure for later usage, and also
    renders the plots in one expander."""
    figs = compute_school_info(df_solution, matching_school_info)
    with st.expander(title, expanded=False):
        for school_id, fig in figs.items():
            st.write(fig["fig_horizontal"])
    return figs


# @st.cache
def compute_school_info(df_solution, matching_school_info):
    from_df = df_solution.groupby(["ncessch_x"]).sum()
    to_df = df_solution.groupby(["ncessch_y"]).sum()
    joined_df = from_df.join(to_df, rsuffix="_new")
    joined_df = joined_df.join(
        matching_school_info.set_index("ncessch"), rsuffix="_info"
    )
    joined_df["Name"] = joined_df["Name"].apply(
        lambda x: select_boxes.pretty_print_school_name(str(x))
    )
    joined_df = joined_df.sort_values(by=["num_total_to_school_new"], ascending=False)

    # Each row is a school, each column is a demographic group.  In each cell
    # is a before-after for the count of students
    row_titles = list(joined_df["Name"])
    row_titles = [r.replace(" Elementary", "") for r in row_titles]
    subplot_titles = list(
        ["_school_name_"]
        + [x + "  (%)" for x in list(config.OUTCOME_TARGETS_DISPLAY.values())]
    ) * len(joined_df)
    # Replace school names
    num_charts_per_school = len(config.OUTCOME_TARGETS_DISPLAY) + 1
    for i, school in enumerate(row_titles):
        subplot_titles[i * num_charts_per_school] = "All " + row_titles[i] + " students"

    school_to_fig = defaultdict(lambda: {})

    for horizontal in [True, False]:
        i = 0
        for ncessch_id, r in joined_df.iterrows():
            school_info = matching_school_info[
                matching_school_info["ncessch"] == ncessch_id
            ]
            fig = make_subplots(
                rows=(horizontal and 1 or num_charts_per_school),
                cols=(horizontal and num_charts_per_school or 1),
                row_titles=horizontal and [row_titles[i]] or None,
                subplot_titles=[
                    "<br>".join(textwrap.wrap(x, 20))
                    for x in subplot_titles[
                        i * num_charts_per_school : (i + 1) * (num_charts_per_school)
                    ]
                ],
                shared_yaxes=False,
            )
            if horizontal:
                # Put school name to the left of row of charts if horizontal
                fig.for_each_annotation(
                    lambda a: a.update(x=-0.05).update(textangle=-90)
                    if a.text in row_titles
                    else ()
                )
            i += 1
            for j, (group, group_display) in enumerate(
                [("total", "All students")]
                + list(config.OUTCOME_TARGETS_DISPLAY.items())
            ):
                # Inconsistency in coding of column name
                col_name = {"hisp": "hispanic"}.get(group, group)
                district_col_name = {
                    "white": "district_perwht",
                    "black": "district_perblk",
                    "native": "district_pernam",
                    "asian": "district_perasn",
                    "total": "district_perblk",
                    "hisp": "district_perhsp",
                }.get(group, "district_per" + group)
                for current, proposed, ideal, visible in [
                    (
                        r["num_" + col_name + "_to_school"],
                        r["num_" + col_name + "_to_school_new"],
                        r["num_" + col_name + "_to_school_new"],
                        (col_name == "total"),
                    ),
                    (
                        (
                            100.0
                            * r["num_" + col_name + "_to_school"]
                            / r["num_total_to_school"]
                        ),
                        (
                            100.0
                            * r["num_" + col_name + "_to_school_new"]
                            / r["num_total_to_school_new"]
                        ),
                        (
                            col_name == "total"
                            and 100.0
                            or (100.0 * school_info[district_col_name].values[0])
                        ),
                        (col_name != "total"),
                    ),
                ]:
                    xs = ["Current", "Proposed"]
                    ys = [current, proposed]
                    if col_name != "total":
                        xs += ["District"]
                        ys += [ideal]
                    fig.add_trace(
                        graph_objects.Bar(
                            x=xs, y=ys, visible=visible, marker_color=config.COLORSCALE
                        ),
                        row=(horizontal and 1 or (j + 1)),
                        col=(horizontal and (j + 1) or 1),
                    )

            fig.update_layout(
                height=horizontal and 200 or 1500,
                width=horizontal and 1000 or 200,
                margin=dict(
                    l=10, r=10, t=50, b=10
                ),  # total width will be 200+10+10=220
                showlegend=False,
            )
            if horizontal:
                school_to_fig[ncessch_id]["fig_horizontal"] = fig
            else:
                school_to_fig[ncessch_id]["fig_vertical"] = fig

    return school_to_fig
