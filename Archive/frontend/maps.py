"""
Renders Current/Proposed boundaries map for the streamlit app
"""

import base64
import os
import random

import folium
import folium.plugins
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config
import select_boxes


def get_from_cache(key):
    fname = os.path.join(config.MAPS_CACHE_DIR, key)
    if os.path.exists(fname):
        return open(fname).read()
    return None


def write_to_cache(key, data):
    fname = os.path.join(config.MAPS_CACHE_DIR, key)
    with open(fname, "w") as fs_out:
        fs_out.write(data)


def write_maps(
        matching_school_info, df_district_shapes, df_solution, map_name_suffix, school_figs
):
    """
    Displays a map centered at the first school in school_info,
    and draws the district attendance zones
    """

    with st.container():
        data = get_from_cache(map_name_suffix)
        spinner_placeholder = st.components.v1.html("", height=100)
        map_placeholder = st.empty()
        if data:
            with spinner_placeholder:
                with st.spinner("Re-rendering district map..."):
                    with map_placeholder:
                        render_map(None, data)
            return

        block_mapper = dict(
            zip(df_solution["block_id"], df_solution["ncessch_y"])
        )  # Map block to new school ID
        with spinner_placeholder:
            with st.spinner("Rendering base map..."):
                m = get_map_base_layer(matching_school_info)
                with map_placeholder:
                    # Since the base map is fast to render, we show it immediately
                    # so that it gives the user something to look forward to while
                    # the final map renders (which can take several seconds)
                    render_map(m, None)
                    # Add search bar after the base map is rendered so that
                    # users don't enter an address until it's finalized.
                    folium.plugins.Geocoder().add_to(m)
            with st.spinner("Rendering proposed district map..."):
                get_map_zone_layer(
                    df_district_shapes, "Proposed", block_mapper, matching_school_info
                ).add_to(m)
            with st.spinner("Rendering current district map..."):
                get_map_zone_layer(
                    df_district_shapes, "Current", None, matching_school_info
                ).add_to(m)
                add_map_markers(m, matching_school_info, school_figs)
            with st.spinner("Rendering district map..."):
                # Fit to bounds of items on map with a bit of margin
                m.fit_bounds(m.get_bounds(), padding=(5, 5))
                # sortLayers sorts by name alphabetically, putting "Current boundaries" first.
                # But the first layer inserted ("Proposed boundaries") will be selected as the default.
                folium.LayerControl(
                    collapsed=False, hideSingleBase=True, sortLayers=True
                ).add_to(m)
                with map_placeholder:
                    data = render_map(m, None)
                    write_to_cache(map_name_suffix, data)
                spinner_placeholder = st.components.v1.html("", height=100)


def school_to_color(ncessch_id):
    """
    Generating random color for each school
    """
    random.seed(ncessch_id)
    return "#" + "%06x" % random.randint(0, 0xFFFFFF)


def color_by_school_id(feature, block_mapper):
    """Choose a random color based on the school ID for the block.  If block_mapper is None,
    use school embedded in the given geo feature; otherwise, look up block ID in block_mapper
    to get the school id to use."""
    if not block_mapper:
        ncessch_id = feature["properties"]["ncessch"]
    else:
        ncessch_id = block_mapper[feature["properties"]["block_id"]]
    return school_to_color(ncessch_id)


def color_by_demographic_fraction(feature, block_mapper):
    return colorscale(block_mapper[feature["properties"]["block_id"]])


def get_map_base_layer(school_info):
    # Center a map at the first school
    lat = school_info.iloc[0]["lat"]
    lng = school_info.iloc[0]["long"]
    m = folium.Map(location=[lat, lng], tiles=None, zoom_start=9, prefer_canvas=True)

    folium.TileLayer(
        show=True,
        overlay=False,
        control=False,
        tiles="CartoDB positron",
        name="Base map",
    ).add_to(m)

    return m


def add_map_markers(m, school_info, school_figs):
    """Note: school_figs is currently unused, but we hope to show them
    when marker is clicked.  Size too big for direct inclusion in fig."""
    for _, r in school_info.iterrows():
        school_name = select_boxes.pretty_print_school_name(r["Name"])
        html = "<b>%s</b>:<b>" % (school_name)

        """
        school_facts = []
        for display, group_code in [("Total enrollment", "enrollment")] + list(
            config.OPTIMIZATION_TARGETS.items()
        ):
            school_facts.append(
                "<b>%s</b>: %d"
                % (display, r["total_" + group_code.replace("hisp", "hispanic")])
            )
        html += "<br>" + "<br>".join(school_facts)
        """

        # Add school-specific plot as an inline image to the marker
        png = school_figs[r["ncessch"]]["fig_vertical"].to_image(format="jpg")
        base64_bytes = base64.b64encode(png)
        data_url = "data:image/jpeg;base64," + base64_bytes.decode("ascii")
        html += '<img src="%s">' % (data_url)

        # The width below should be a little greater than the width of the impact plots
        # in "vertical" mode (see bottom of impact_plots.py)
        iframe = folium.IFrame(html=html, height=400, width=230)

        # TODO: this popup renders above where the user clicks, but ideally it would be
        # rendered below, so that the dismiss X is easy to access.  Not sure how to do that here.
        popup = folium.Popup(iframe)

        folium.Marker(
            [r["lat"], r["long"]],
            popup=popup,
            icon=folium.Icon(icon_color=school_to_color(r["ncessch"])),
        ).add_to(m)
    return m


def hash_geojson_obj(geojson_obj):
    return hash(str(geojson_obj))


def format_block_tooltip(block_row, new_school_id, ncessch_to_name):
    s = "Census block *" + str(block_row["block_id"])[-6:]  # Is this useful?

    old_school_name = select_boxes.pretty_print_school_name(
        ncessch_to_name[block_row["ncessch"]]
    )
    if new_school_id:
        new_school_name = select_boxes.pretty_print_school_name(
            ncessch_to_name[new_school_id]
        )
        if old_school_name != new_school_name:
            return s + " REZONED from %s to %s" % (old_school_name, new_school_name)
        else:
            return s + " still mapped to %s" % (new_school_name)
    return s + " currently mapped to %s" % (old_school_name)


# NOTE:  Ever since Streamlit changed their caching decorator, this method
# cannot be cached without extra work, since _df_district_shapes is not cacheable
# and there's no longer a way to specify how to treat it.  Follow
# https://github.com/streamlit/streamlit/issues/6295 for more info
def get_map_zone_layer(_df_district_shapes, name, block_mapper, school_info):
    df_shapes = _df_district_shapes.copy()
    ncessch_to_name = pd.Series(
        school_info["Name"].values, index=school_info["ncessch"]
    ).to_dict()

    if name == "Current":
        df_shapes["BlockTooltip"] = df_shapes.apply(
            lambda x: format_block_tooltip(x, None, ncessch_to_name), axis=1
        )
        sfn = lambda feature: {
            "fillColor": color_by_school_id(feature, None),
            "fillOpacity": 0.7,
            "stroke": False,
            "weight": 1,
        }
    else:
        df_shapes["BlockTooltip"] = df_shapes.apply(
            lambda x: format_block_tooltip(
                x, block_mapper[x["block_id"]], ncessch_to_name
            ),
            axis=1,
        )
        sfn = lambda feature: {
            "fillColor": color_by_school_id(feature, block_mapper),
            "fillOpacity": 0.7,
            #            "fillOpacity": ("REZONED" in feature["properties"]["BlockTooltip"]) and 0.6 or 0.7,
            #            "dashArray": ("REZONED" in feature["properties"]["BlockTooltip"]) and "2" or None,
            #            "stroke": "REZONED" in feature["properties"]["BlockTooltip"],
            "stroke": False,
            "weight": 1,
        }
    layer = folium.GeoJson(
        df_shapes,
        overlay=False,
        style_function=sfn,
        name=name + " boundaries",
        tooltip=folium.features.GeoJsonTooltip(fields=["BlockTooltip"], labels=False),
        highlight_function=lambda x: {
            "fillColor": "#000000",
            "color": "#000000",
            "fillOpacity": 0.50,
            "weight": 0.1,
        },
    )

    return layer


def render_map(fig, fig_data, height=500):
    if not fig_data:
        fig = folium.Figure().add_child(fig)
        fig_data = fig.render()
    components.html(fig_data, height=height + 10)
    return fig_data
