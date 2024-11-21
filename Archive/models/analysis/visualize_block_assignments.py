from utils.header import *
import geopandas as gpd


def viz_assignments(
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
    state_blocks,
    pre_solver_file="models/solver_files/{}/{}/prepped_file_for_solver_{}.csv",
    # solution_dir="simulation_outputs/2122_top_100_norm_exp/{}/{}/{}/{}_{}_{}_{}_{}_{}/",
    solution_dir="models/results/{}/{}/{}/{}_{}_{}_{}_{}_{}/",
    save_file=True,
):

    if state_blocks.empty:
        state_codes_file = "data/state_codes.csv"
        blocks_shape_file = "data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp"
        df_states = pd.read_csv(state_codes_file, dtype=str)
        state_fips = df_states[df_states["abbrev"] == state].iloc[0]["fips_code"]
        state_blocks = gpd.read_file(blocks_shape_file.format(state_fips, state_fips))
        state_blocks["GEOID20"] = state_blocks["GEOID20"].astype(int)

    df_asgn_orig = pd.read_csv(pre_solver_file.format(year, state, district_id))
    df_asgn_new = pd.read_csv(
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

    df_orig = gpd.GeoDataFrame(
        pd.merge(
            df_asgn_orig,
            state_blocks,
            left_on="block_id",
            right_on="GEOID20",
            how="inner",
        )
    )

    df_new = gpd.GeoDataFrame(
        pd.merge(
            df_asgn_new,
            state_blocks,
            left_on="block_id",
            right_on="GEOID20",
            how="inner",
        )
    )

    _, (ax1) = plt.subplots(1, 1, figsize=(60, 40))
    district_map_orig = df_orig.boundary.plot(ax=ax1)
    df_schools = df_orig.groupby("ncessch", as_index=False).agg(
        {"lat": "first", "long": "first"}
    )
    df_schools["geometry"] = gpd.points_from_xy(df_schools.long, df_schools.lat)
    df_schools = df_schools.set_geometry("geometry")
    all_schools_nces = df_schools["ncessch"].tolist()

    # Generate colors
    colors = {}
    for nces in all_schools_nces:
        random.seed(int(nces))
        colors[nces] = "#" + "%06x" % random.randint(0, 0xFFFFFF)

    # Original assignments
    for nces in all_schools_nces:
        curr = df_orig[df_orig["ncessch"].isin([nces])]
        curr.plot(ax=district_map_orig, color=colors[nces], zorder=1)
    df_schools.plot(ax=ax1, marker="P", color="black", zorder=2)
    output_file = (
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
        + "original_zoning.png"
    )
    if save_file:
        plt.axis("off")
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    # New assignments
    district_map_new = df_new.boundary.plot(ax=ax1)

    for nces in all_schools_nces:
        curr = df_new[df_new["new_school_nces"].isin([nces])]
        curr.plot(ax=district_map_new, color=colors[nces], zorder=1)

    # # Attempting to make changes more interpretable
    # for nces in all_schools_nces:
    #     curr = df_new[df_new["new_school_nces"].isin([nces])].reset_index(drop=True)
    #     colors_to_plot = []
    #     for i in range(0, len(curr)):
    #         if curr["new_school_nces"][i] == curr["ncessch_x"][i]:
    #             colors_to_plot.append("lightgray")
    #         else:
    #             colors_to_plot.append(colors[nces])
    #     curr.plot(ax=district_map_new, color=colors_to_plot, zorder=1)

    df_schools.plot(ax=ax1, marker="P", color="black", zorder=2)
    output_file = (
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
        + "rezoning.png"
    )
    if save_file:
        plt.axis("off")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def viz_orig_assignments_and_demos(
    district_id,
    year,
    state,
    max_percent_distance_increase,
    max_percent_size_increase,
    optimizing_for,
    objective_function,
    percent_neighbors_rezoned_together,
    enforce_contiguity,
    state_blocks,
    demos_key="num_white",
    drop_empty_blocks=True,
    blocks_demos_file="data/derived_data/{}/{}/full_blocks_data.csv",
    pre_solver_file="models/solver_files/{}/{}/prepped_file_for_solver_{}.csv",
    output_file="simulation_outputs/2122_all_usa_dissim/{}/{}/{}/{}_{}_{}_{}_{}_{}/{}.png",
):

    df_asgn_orig = pd.read_csv(pre_solver_file.format(year, state, district_id))
    if drop_empty_blocks:
        df_asgn_orig = df_asgn_orig[
            df_asgn_orig["num_total_to_school"] > 0
        ].reset_index(drop=True)
    blocks_and_districts = df_asgn_orig[["block_id", "leaid"]]

    print("Loading block demos ...")
    df_block_demos = pd.merge(
        pd.read_csv(blocks_demos_file.format(year, state)),
        blocks_and_districts,
        on="block_id",
        how="inner",
    )
    df_block_demos["cat_percent_to_viz"] = (
        df_block_demos[demos_key] / df_block_demos["num_total"]
    )

    n_bins = 100
    df_block_demos["cat_percentile_bucket"] = pd.cut(
        df_block_demos["cat_percent_to_viz"],
        bins=n_bins,
        precision=1,
        labels=list(range(0, n_bins)),
    )

    print("Creating geodataframes ...")
    df_orig = gpd.GeoDataFrame(
        pd.merge(
            df_asgn_orig,
            state_blocks,
            left_on="block_id",
            right_on="GEOID20",
            how="inner",
        )
    )
    df_demos = gpd.GeoDataFrame(
        pd.merge(
            df_block_demos,
            state_blocks,
            left_on="block_id",
            right_on="GEOID20",
            how="inner",
        )
    )

    # Constrain to geography of attendance boundary
    df_demos = df_demos[
        df_demos["block_id"].isin(df_orig["block_id"].tolist())
    ].reset_index(drop=True)

    _, (ax1) = plt.subplots(1, 1, figsize=(60, 40))
    district_demos_map = df_demos.boundary.plot(ax=ax1)
    df_schools = df_orig.groupby("ncessch", as_index=False).agg(
        {"lat": "first", "long": "first"}
    )
    df_schools["geometry"] = gpd.points_from_xy(df_schools.long, df_schools.lat)
    df_schools = df_schools.set_geometry("geometry")
    all_schools_nces = df_schools["ncessch"].tolist()

    # Generate colors for schools
    print("Getting colors for schools ...")
    school_colors = {}
    for nces in all_schools_nces:
        random.seed(int(nces))
        school_colors[nces] = "#" + "%06x" % random.randint(0, 0xFFFFFF)

    # Generate color gradient for demos
    print("Getting colors for demos ...")
    buckets = list(set(df_block_demos["cat_percentile_bucket"].tolist()))
    demos_colors = ["" for i in range(0, len(buckets))]
    step = int(100 / len(buckets))
    base_color_hex = "#0000FF"
    for i, b in enumerate(buckets):
        alpha = str(int(step) * i)
        if len(alpha) == 1:
            alpha = "0" + alpha
        demos_colors[i] = base_color_hex + alpha

    print("Plotting demos ...")
    # Demographics
    for i, b in enumerate(buckets):
        curr = df_demos[df_demos["cat_percentile_bucket"].isin([b])]
        curr.plot(
            ax=district_demos_map,
            color=demos_colors[i],
            zorder=1,
        )
    df_schools.plot(ax=ax1, marker="P", color="black", zorder=2)

    # plt.show()
    output_file = output_file.format(
        year,
        state,
        district_id,
        max_percent_distance_increase,
        max_percent_size_increase,
        optimizing_for,
        objective_function,
        percent_neighbors_rezoned_together,
        enforce_contiguity,
        demos_key,
    )
    plt.axis("off")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def output_viz_for_campaign(
    input_dir="simulation_outputs/2122_all_usa_dissim",
    districts_file="data/school_covariates/top_0_1000_largest_districts.csv",
    state_codes_file="data/state_codes.csv",
    blocks_shape_file="data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp",
):
    df_states = pd.read_csv(state_codes_file, dtype=str)
    df_dists = pd.read_csv(districts_file, dtype=str)
    year = "2122"
    input_dir = os.path.join(input_dir, year)
    states = os.listdir(input_dir)
    states.reverse()
    for state in states:
        state_fips = df_states[df_states["abbrev"] == state].iloc[0]["fips_code"]
        print("Loading blocks data for ...", state)
        state_blocks = gpd.read_file(blocks_shape_file.format(state_fips, state_fips))
        state_blocks["GEOID20"] = state_blocks["GEOID20"].astype(int)
        curr_dists_in_folder = os.listdir(os.path.join(input_dir, state))
        curr_dists = set(df_dists["leaid"].tolist()).intersection(curr_dists_in_folder)
        for district_id in curr_dists:
            for config in os.listdir(os.path.join(input_dir, state, district_id)):
                print(state, district_id, config)
                config_params = config.split("_")
                max_percent_distance_increase = config_params[0]
                max_percent_size_increase = config_params[1]
                optimizing_for = config_params[2]
                objective_function = "_".join(config_params[3:6])
                percent_neighbors_rezoned_together = config_params[6]
                enforce_contiguity = config_params[7]
                solution_file = ""
                already_made_images = False
                if not (
                    enforce_contiguity == "True"
                    and percent_neighbors_rezoned_together == "0.5"
                    and max_percent_size_increase in ["0.1", "0.15"]
                    and max_percent_distance_increase == "0.5"
                    and objective_function == "min_total_segregation"
                ):
                    continue

                for f in os.listdir(
                    os.path.join(input_dir, state, district_id, config)
                ):
                    if f.startswith("solution"):
                        solution_file = f
                    if f.endswith(".png"):
                        already_made_images = True

                if not already_made_images:
                    viz_assignments(
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
                        state_blocks,
                    )

                # viz_orig_assignments_and_demos(
                #     district_id,
                #     year,
                #     state,
                #     max_percent_distance_increase,
                #     max_percent_size_increase,
                #     optimizing_for,
                #     objective_function,
                #     percent_neighbors_rezoned_together,
                #     enforce_contiguity,
                #     state_blocks,
                #     demos_key="num_white",
                # )


if __name__ == "__main__":

    state_codes_file = "data/state_codes.csv"
    blocks_shape_file = "data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp"
    district_id = "3702970"
    year = "2122"
    state = "NC"
    solution_file = "solution_1.1288_19800.480862350003_2_1583544_367694.csv"
    max_percent_distance_increase = 0.5
    max_percent_size_increase = 0.15
    optimizing_for = "white"
    objective_function = "min_total_segregation"
    percent_neighbors_rezoned_together = 0.5
    enforce_contiguity = True

    df_states = pd.read_csv(state_codes_file, dtype=str)
    df_states = df_states[df_states["abbrev"].isin([state])].reset_index()
    state_fips = df_states["fips_code"][0]
    print("Loading blocks data ...")
    state_blocks = gpd.read_file(blocks_shape_file.format(state_fips, state_fips))
    state_blocks["GEOID20"] = state_blocks["GEOID20"].astype(int)

    # viz_assignments(
    #     district_id,
    #     year,
    #     state,
    #     solution_file,
    #     max_percent_distance_increase,
    #     max_percent_size_increase,
    #     optimizing_for,
    #     objective_function,
    #     percent_neighbors_rezoned_together,
    #     "True" if enforce_contiguity else "False",
    #     state_blocks,
    # )

    viz_orig_assignments_and_demos(
        district_id,
        year,
        state,
        max_percent_distance_increase,
        max_percent_size_increase,
        optimizing_for,
        objective_function,
        percent_neighbors_rezoned_together,
        "True" if enforce_contiguity else "False",
        state_blocks,
        demos_key="num_frl",
    )

    # output_viz_for_campaign()
