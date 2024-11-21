#!/usr/bin/env python3

"""
Generates shape file (geopandas csv dumps) per district for a given state
"""

import pandas as pd

import folium
import geopandas as gpd
import os

# Set up the location of all the static data.
DATA_DIR = "../simulation_outputs/"
# DATA_DIR = ""


if __name__ == "__main__":

    year = "2122"
    df_states = pd.read_csv("data/state_codes.csv", dtype=str)
    df_states = df_states[df_states["abbrev"].isin(["VA"])].reset_index(drop=True)
    for i in range(0, len(df_states)):
        state = df_states["abbrev"][i]
        state_fips = df_states["fips_code"][i]

        output_dir = DATA_DIR + (
            "data/census_block_shapefiles_2020/%s-%s" % (year, state)
        )

        print("Processing {}...".format(state))

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir, exist_ok=True)
        # else:
        #     continue

        filename_school_state_data = (
            DATA_DIR + "data/derived_data/2122/%s/schools_file_for_assignment.csv"
        )
        filename_pre_solver_file = (
            DATA_DIR + "models/solver_files/{}/{}/prepped_file_for_solver_{}.csv"
        )

        try:
            df_school_state_data = pd.read_csv(
                filename_school_state_data % (state), dtype=str
            )
        except Exception as e:
            continue

        districts = list(df_school_state_data["leaid"].unique())
        filename_blocks_shape_file = (
            DATA_DIR
            + "data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp"
        )

        blocks = gpd.read_file(
            filename_blocks_shape_file.format(state_fips, state_fips)
        )

        for district in districts:
            fname = output_dir + "/%s-%s-%s.geodata.csv" % (year, state, district)
            # if os.path.exists(fname):
            #     continue
            print(district)
            blocks["GEOID20"] = blocks["GEOID20"].astype(int)
            df_asgn_orig = pd.read_csv(
                filename_pre_solver_file.format(year, state, district)
            )
            df_orig = gpd.GeoDataFrame(
                pd.merge(
                    df_asgn_orig,
                    blocks,
                    left_on="block_id",
                    right_on="GEOID20",
                    how="inner",
                )
            )
            df_orig.crs = "epsg:4326"
            df_orig["geometry"] = df_orig["geometry"].simplify(0.001)

            print(fname)
            df_orig.to_csv(fname)
