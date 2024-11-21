from numpy.core.shape_base import block
from pandas.core.internals import blocks
from utils.header import *
import geopandas as gpd
import haversine as hs
from utils.distances_and_times import compute_travel_info_to_schools


def output_block_to_school_mapping(
    states_codes_file="data/state_codes.csv",
    block_shapes_dir="data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/",
    block_shapes_file_name="tl_2021_{}_tabblock20.shp",
    sab_elem_file="data/school_boundary_shapefiles/ATTOM_boundaries_21_22_USA/school-attendance-areas-03.shp",
    output_dir="data/derived_data/{}/",
    output_file_name="blocks_to_elementary.csv",
    states=["VA"],
    zones_years=["2122"],
):

    df_states = pd.read_csv(states_codes_file).astype(str)
    for y in zones_years:
        print("Loading elementary school zones for year {}...".format(y, y))
        elem_zones = (
            gpd.read_file(sab_elem_file.format(y, y))
            .to_crs(epsg=3857)
            .rename(
                columns={
                    "NCESDISTID": "leaid",
                    "NCESSCHID": "ncessch",
                    "OPENENROLL": "openEnroll",
                    "AREASQMI": "Shape_Area",
                }
            )
        )

        # # Check how many closed enrollment schools we have in VA, boundary overlaps
        # va_zones = elem_zones[elem_zones["leaid"].str.startswith("51")]
        # va_zones_ce = va_zones[va_zones["openEnroll"] == "N"]
        # print(len(va_zones), len(va_zones_ce), len(va_zones_ce) / len(va_zones))
        # overlap_va = gpd.sjoin(va_zones_ce, va_zones_ce, how="inner", op="intersects")
        # print(len(va_zones_ce), len(overlap_va))
        # exit()

        print(
            "Removing schools that are not completely closed enrollment — i.e. attendance defined by zoned geographies ..."
        )
        elem_zones = elem_zones[elem_zones["openEnroll"] == "N"].reset_index()
        # Make the output dir for this year of attendance boundaries if it doesn't already exist
        if not os.path.exists(output_dir.format(y)):
            os.mkdir(output_dir.format(y))

        # Only process those states we haven't processed yet
        states_already_computed = [f for f in os.listdir(output_dir.format(y))]
        df_states = df_states[
            ~df_states["abbrev"].isin(states_already_computed)
        ].reset_index()
        # df_states = df_states[df_states["abbrev"].isin(states)].reset_index()

        for i in range(0, len(df_states)):
            try:
                state_code = str(df_states["fips_code"][i])
                state_abbrev = df_states["abbrev"][i]
                if len(state_code) < 2:
                    state_code = "0" + state_code
                print(
                    "Loading blocks for state {} {} ...".format(
                        state_code, state_abbrev
                    )
                )
                curr_dir = block_shapes_dir.format(state_code)
                curr_file = block_shapes_file_name.format(state_code)
                curr_blocks = gpd.read_file(curr_dir + curr_file)
                print("\tProjecting to elem zones crs ...")
                curr_blocks = curr_blocks.to_crs(elem_zones.crs)
                print("\tSetting centroids and updating geometry column ...")
                curr_blocks["geometry"] = curr_blocks.centroid
                curr_blocks = curr_blocks.set_geometry("geometry")
                print("\tChecking which blocks are contained in which zones ...")
                blocks_within_zones = gpd.sjoin(
                    curr_blocks, elem_zones, how="inner", op="intersects"
                )

                print("\tGetting lat long for block centroids ...")
                blocks_within_zones = blocks_within_zones.to_crs(epsg=4326)
                blocks_within_zones["block_centroid_lat"] = blocks_within_zones[
                    "geometry"
                ].y
                blocks_within_zones["block_centroid_long"] = blocks_within_zones[
                    "geometry"
                ].x

                curr_output_dir = output_dir.format(y) + state_abbrev + "/"
                if not os.path.exists(curr_output_dir):
                    os.mkdir(curr_output_dir)

                print("\tSorting by shape area and dropping duplicates ...")
                blocks_within_zones.sort_values(
                    by="Shape_Area", ascending=True, inplace=True
                )

                # print(len(blocks_within_zones))
                blocks_within_zones = blocks_within_zones.drop_duplicates(
                    subset=["GEOID20"]
                )
                # print(len(blocks_within_zones))
                # exit()
                print(
                    "\tOutputting merged file to file {} ...".format(
                        curr_output_dir + output_file_name
                    )
                )
                blocks_within_zones.to_csv(
                    curr_output_dir + output_file_name, index=False
                )

                # Delete dataframes
                del curr_blocks
                del blocks_within_zones

            except Exception as e:
                print(e)
                continue


def estimate_num_ell_per_block(df_demos, bgrp_language_data_file):
    df_language = pd.read_csv(bgrp_language_data_file, encoding="ISO-8859-1", dtype=str)
    df_language["block_group_id"] = df_language[
        ["STATEA", "COUNTYA", "TRACTA", "BLKGRPA"]
    ].agg("".join, axis=1)

    # df_language["est_ell_frac"] = (
    #     pd.to_numeric(df_language["ALWUE007"])
    #     + pd.to_numeric(df_language["ALWUE008"])
    #     + pd.to_numeric(df_language["ALWUE012"])
    #     + pd.to_numeric(df_language["ALWUE013"])
    #     + pd.to_numeric(df_language["ALWUE017"])
    #     + pd.to_numeric(df_language["ALWUE018"])
    #     + pd.to_numeric(df_language["ALWUE022"])
    #     + pd.to_numeric(df_language["ALWUE023"])
    # ) / pd.to_numeric(df_language["ALWUE002"])

    # Estimate ELL population, very coarsely, as number of
    # 5-17 year olds who speak anything but just English at home
    # NOTE: this will be re-scaled when we estimate the number
    # of students from each block and category zoned to each school
    df_language["est_ell_frac"] = (
        pd.to_numeric(df_language["ALWUE002"]) - pd.to_numeric(df_language["ALWUE003"])
    ) / pd.to_numeric(df_language["ALWUE002"])
    df_language = df_language[["block_group_id", "est_ell_frac"]]
    df_merged = pd.merge(df_demos, df_language, how="left", on="block_group_id")
    num_ell = []
    for i in range(0, len(df_merged)):
        num_ell.append(
            np.round(df_merged["num_total"][i] * df_merged["est_ell_frac"][i])
        )
    return num_ell


def estimate_num_frl_per_block(df_demos, bgrp_poverty_data_file):
    df_poverty = pd.read_csv(bgrp_poverty_data_file, encoding="ISO-8859-1", dtype=str)
    df_poverty["block_group_id"] = df_poverty[
        ["STATEA", "COUNTYA", "TRACTA", "BLKGRPA"]
    ].agg("".join, axis=1)

    # Estimate FRL population as % of population in each block
    # that has income 1.85x or below the Federal poverty threshold
    df_poverty["est_frl_frac"] = (
        pd.to_numeric(df_poverty["ALWVE001"])
        - pd.to_numeric(df_poverty["ALWVE007"])
        - pd.to_numeric(df_poverty["ALWVE008"])
    ) / pd.to_numeric(df_poverty["ALWVE001"])
    df_poverty = df_poverty[["block_group_id", "est_frl_frac"]]
    df_merged = pd.merge(df_demos, df_poverty, how="left", on="block_group_id")
    num_frl = []
    for i in range(0, len(df_merged)):
        num_frl.append(
            np.round(df_merged["num_total"][i] * df_merged["est_frl_frac"][i])
        )
    return num_frl


def load_and_prep_block_racial_demos(
    block_demos_all_file,
    block_demos_over_18_file,
    bgrp_ell_file_name,
    bgrp_frl_file_name,
    census_block_mapping_file="data/census_block_covariates/census_block_mapping_2020_2010.csv",
):

    print("\tLoad mapping file ...")
    # Map a given 2020 block to its highest-weighted block from 2010
    df_mapping = (
        pd.read_csv(census_block_mapping_file, encoding="ISO-8859-1", dtype=str)
        .sort_values(by="WEIGHT", ascending=False)
        .groupby(["GEOID20"])
        .agg({"GEOID10": "first"})
    )

    # Start with block demographics because these don't change regardless of the year of the attendance boundaries we are considering
    print("\tLoading block demos ...")
    df_block_demos_all = pd.read_csv(
        block_demos_all_file, encoding="ISO-8859-1", dtype=str
    )
    df_block_demos_over_18 = pd.read_csv(
        block_demos_over_18_file, encoding="ISO-8859-1", dtype=str
    )

    print("\tCreating block group ID and block ID fields ...")

    df_block_demos_all["block_id"] = df_block_demos_all[
        ["STATEA", "COUNTYA", "TRACTA", "BLOCKA"]
    ].agg("".join, axis=1)

    df_block_demos_all = pd.merge(
        df_block_demos_all,
        df_mapping,
        left_on="block_id",
        right_on="GEOID20",
        how="inner",
    )

    df_block_demos_all["block_group_id"] = df_block_demos_all["GEOID10"].apply(
        lambda x: x[:-3]
    )

    df_block_demos_over_18["block_id"] = df_block_demos_over_18[
        ["STATEA", "COUNTYA", "TRACTA", "BLOCKA"]
    ].agg("".join, axis=1)

    print(
        "\tStore number of kids under 18 (proxy for school-going aged children) per race, per block..."
    )

    df_block_demos = pd.merge(
        df_block_demos_all, df_block_demos_over_18, on="block_id", how="inner"
    )

    df_block_demos["num_total"] = pd.to_numeric(
        df_block_demos_all["U7B001"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D001"])

    df_block_demos["num_white"] = pd.to_numeric(
        df_block_demos_all["U7B003"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D003"])

    df_block_demos["num_black"] = pd.to_numeric(
        df_block_demos_all["U7B004"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D004"])

    df_block_demos["num_native"] = pd.to_numeric(
        df_block_demos_all["U7B005"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D005"])

    df_block_demos["num_asian"] = pd.to_numeric(
        df_block_demos_all["U7B006"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D006"])

    df_block_demos["num_hispanic"] = pd.to_numeric(
        df_block_demos_all["U7C002"]
    ) - pd.to_numeric(df_block_demos_over_18["U7E002"])

    print(
        "\tEstimate block group / block FRL population (using block group income and FRL income cutoffs) ..."
    )
    df_block_demos["num_ell"] = estimate_num_ell_per_block(
        df_block_demos, bgrp_ell_file_name
    )
    df_block_demos["num_frl"] = estimate_num_frl_per_block(
        df_block_demos, bgrp_frl_file_name
    )

    return df_block_demos


def load_and_prep_school_demos(
    state,
    school_covars_file="data/school_covariates/ccd_1920_school_covars_by_state/{}/ccd_1920_crdc_1718_school_covars.csv",
    school_locs_file="data/school_covariates/nces_21_22_lat_longs.csv",
):

    df_s = pd.read_csv(school_covars_file.format(state), dtype=str)

    df_s["totenrl"] = pd.to_numeric(df_s["total_students"])
    df_s["perwht"] = pd.to_numeric(df_s["num_white"]) / df_s["totenrl"]
    df_s["perblk"] = pd.to_numeric(df_s["num_black"]) / df_s["totenrl"]
    df_s["pernam"] = pd.to_numeric(df_s["num_native"]) / df_s["totenrl"]
    df_s["perasn"] = pd.to_numeric(df_s["num_asian"]) / df_s["totenrl"]
    df_s["perhsp"] = pd.to_numeric(df_s["num_hispanic"]) / df_s["totenrl"]
    df_s["perfrl"] = pd.to_numeric(df_s["num_frl"]) / df_s["totenrl"]
    df_s["perell"] = pd.to_numeric(df_s["num_ell"]) / df_s["totenrl"]

    df_s = df_s[
        [
            "NCESSCH",
            "totenrl",
            "perwht",
            "perblk",
            "pernam",
            "perasn",
            "perhsp",
            "perfrl",
            "perell",
        ]
    ]

    # Add in lat longs for schools
    df_school_locs = pd.read_csv(school_locs_file, dtype=str)
    df_school_locs.dropna(subset=["lat", "long"], inplace=True)
    df_sch_covars = pd.merge(
        df_s, df_school_locs, left_on="NCESSCH", right_on="nces_id", how="inner"
    )
    return df_sch_covars


def add_in_demos_to_block_data(
    state,
    df_sch_covars,
    input_dir="data/derived_data/2122/",
    block_demos_all_file_name="data/census_block_covariates/2020_census_race_hisp_all_by_block/by_state/{}/racial_demos_by_block.csv",
    block_demos_over_18_file_name="data/census_block_covariates/2020_census_race_hisp_over_18_by_block/by_state/{}/racial_demos_by_block.csv",
    bgrp_ell_file_name="data/census_block_covariates/2015_2019_ACS_block_group_language/by_state/{}/ell_demos_by_block_group.csv",
    bgrp_frl_file_name="data/census_block_covariates/2015_2019_ACS_block_group_poverty_ratios/by_state/{}/frl_demos_by_block_group.csv",
    blocks_to_schools_file="blocks_to_elementary.csv",
    output_file="data/derived_data/2122/{}/full_blocks_data.csv",
):

    print("\tLoading block demos for state {}...".format(state))
    df_block_demos = load_and_prep_block_racial_demos(
        block_demos_all_file_name.format(state),
        block_demos_over_18_file_name.format(state),
        bgrp_ell_file_name.format(state),
        bgrp_frl_file_name.format(state),
    )

    print("\tLoading zoned blocks for state {} ...".format(state))
    curr_dir = input_dir + state + "/"
    df_block_zones = pd.read_csv(curr_dir + blocks_to_schools_file, dtype=str)
    print("\tJoining zoned blocks on school covars ...")
    df = pd.merge(
        df_block_zones,
        df_sch_covars,
        left_on="ncessch",
        right_on="NCESSCH",
        how="left",
    )
    df.dropna(
        subset=["block_centroid_lat", "block_centroid_long", "lat", "long"],
        inplace=True,
    )

    print("\tJoining zoned blocks and school covars on block demos ...")
    df = pd.merge(
        df, df_block_demos, left_on="GEOID20", right_on="block_id", how="inner"
    )

    print("\tComputing travel info to school ...")
    distances, travel_times, directions = compute_travel_info_to_schools(df)
    df["distance_to_school"] = distances
    df["travel_time_to_school"] = travel_times
    df["directions_to_school"] = directions

    # df["distance_to_school"] = [0 for i in range(0, len(df))]
    # df["travel_time_to_school"] = [0 for i in range(0, len(df))]
    # df["directions_to_school"] = [0 for i in range(0, len(df))]

    print("\tSubsetting columns and outputting csv for {}".format(state))
    df = df[
        [
            "block_id",
            "block_group_id",
            "block_centroid_lat",
            "block_centroid_long",
            "distance_to_school",
            "travel_time_to_school",
            "directions_to_school",
            "num_total",
            "num_white",
            "num_black",
            "num_native",
            "num_asian",
            "num_hispanic",
            "num_ell",
            "num_frl",
            "ncessch",
            "leaid",
            "lat",
            "long",
            "openEnroll",
            "totenrl",
            "perwht",
            "pernam",
            "perasn",
            "perhsp",
            "perblk",
            "perfrl",
            "perell",
        ]
    ]

    df.to_csv(output_file.format(state), index=False)
    del df
    del df_block_zones


def add_in_demos_to_block_data_parallel(
    input_dir="data/derived_data/2122/",
    processing_function=add_in_demos_to_block_data,
):

    N_THREADS = 10

    print("Load and process school-level demographics ...")

    curr_states = os.listdir(input_dir)
    all_states_to_process = []
    # curr_states = ["NC"]
    for state in curr_states:
        print(state)
        df_sch_covars_state = load_and_prep_school_demos(state)
        all_states_to_process.append((state, df_sch_covars_state))

    print("Starting multiprocessing ...")
    print(len(all_states_to_process))
    from multiprocessing import Pool

    p = Pool(N_THREADS)
    p.starmap(processing_function, all_states_to_process)

    p.terminate()
    p.join()


if __name__ == "__main__":
    output_block_to_school_mapping()
    # add_in_demos_to_block_data()
    add_in_demos_to_block_data_parallel()
