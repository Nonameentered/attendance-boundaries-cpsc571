# visualize.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


def plot_racial_composition(df_results, school_compositions):
    schools = list(school_compositions.keys())
    white_percents = [school_compositions[school]["White%"] * 100 for school in schools]
    black_percents = [school_compositions[school]["Black%"] * 100 for school in schools]
    asian_percents = [school_compositions[school]["Asian%"] * 100 for school in schools]

    fig = plt.figure(figsize=(12, 6))
    x = range(len(schools))
    width = 0.25

    plt.bar([p - width for p in x], white_percents, width, label="White%")
    plt.bar(x, black_percents, width, label="Black%")
    plt.bar([p + width for p in x], asian_percents, width, label="Asian%")

    plt.xlabel("Schools")
    plt.ylabel("Percentage")
    plt.title("Racial Composition at Each School")
    plt.xticks(x, schools, rotation=45)
    plt.legend()
    plt.tight_layout()

    return fig


def plot_school_entropies(school_entropies):
    schools = list(school_entropies.keys())
    entropies = [school_entropies[school] for school in schools]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(schools, entropies, color="skyblue")
    plt.xlabel("Schools")
    plt.ylabel("Entropy (Theil Index)")
    plt.title("Entropy at Each School")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_map(df_results, gdf_schools):
    gdf_results = gpd.GeoDataFrame(df_results, geometry="geometry")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    gdf_results.plot(column="School", ax=ax, legend=True, cmap="tab20")
    gdf_schools.plot(ax=ax, color="red", markersize=50)
    plt.title("Map of Tracts Assigned to Schools")
    plt.axis("off")

    return fig


def plot_interactive_map(df_results, gdf_schools):
    # Ensure df_results is a GeoDataFrame with CRS set
    gdf_results = gpd.GeoDataFrame(df_results, geometry="geometry")

    # Set CRS if not already set
    if gdf_results.crs is None:
        gdf_results.set_crs(epsg=4326, inplace=True)

    # Ensure schools GeoDataFrame has CRS set
    if gdf_schools.crs is None:
        gdf_schools.set_crs(epsg=4326, inplace=True)

    # Transform to WGS84 for Plotly (lat/lon)
    gdf_results = gdf_results.to_crs(epsg=4326)
    gdf_schools = gdf_schools.to_crs(epsg=4326)

    # Calculate centroids in lat/lon
    gdf_results["centroid"] = gdf_results["geometry"].centroid

    # Prepare data for Plotly
    df_plot = gdf_results.copy()
    df_plot["lon"] = df_plot["centroid"].x
    df_plot["lat"] = df_plot["centroid"].y

    # Create Scatter Mapbox for Tracts
    fig = px.scatter_mapbox(
        df_plot,
        lat="lat",
        lon="lon",
        hover_name="Tract",
        color_discrete_sequence=["blue"],  # All tract markers are blue
        opacity=0.6,
        zoom=10,
        height=600,
    )

    # Add School Locations
    schools_plot = gdf_schools.copy()
    schools_plot["lon"] = schools_plot["geometry"].x
    schools_plot["lat"] = schools_plot["geometry"].y

    # Add school points manually using graph_objects
    for i, row in schools_plot.iterrows():
        fig.add_scattermapbox(
            lat=[row["lat"]],
            lon=[row["lon"]],
            mode="markers",
            marker=dict(size=15, color="red"),  # Customize size and color here
            name=row["Name"],  # Use the school name as legend/tooltip
        )

    # Update Layout
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig
