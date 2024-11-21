# -*- coding: utf-8 -*-
# from networkx.classes.function import all_neighbors
from utils.header import *
import haversine as hs
import geopandas as gpd
from networkx.readwrite import json_graph
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from utils.distances_and_times import (
    get_distance_for_coord_pair,
    get_travel_time_for_coord_pair,
    impute_vals_two_d,
)


def output_solver_solution(
    x,
    solver,
    is_cp,
    objective_value,
    time_ellapsed,
    num_iterations,
    num_bb_nodes,
    optimal,
    optimizing_for,
    df,
    district_id,
    year,
    state,
    max_percent_distance_increase,
    max_percent_size_increase,
    objective_function,
    percent_neighbors_rezoned_together,
    enforce_contiguity,
    input_file_schools="data/derived_data/{}/{}/schools_file_for_assignment.csv",
    input_file_blocks="data/derived_data/{}/{}/blocks_file_for_assignment.csv",
    output_dir="models/results/{}/{}/{}/{}_{}_{}_{}_{}_{}/",
    output_file="solution_{}_{}_{}_{}_{}.csv",
    s3_output_dir="",
):

    df_schools = pd.read_csv(input_file_schools.format(year, state))
    df_blocks = pd.read_csv(input_file_blocks.format(year, state))

    block_idx = set(df["block_idx"].tolist())
    school_idx = set(df["school_idx"].tolist())

    solution_data = {
        "block_id": [],
        "new_school_nces": [],
        "new_distance_to_school": [],
    }

    for i in block_idx:
        for j in school_idx:
            condition_val = False
            if is_cp:
                condition_val = solver.BooleanValue(x[i][j])
            else:
                condition_val = x[i][j].solution_value() > 0.5
            if condition_val:

                init_block_df = df[df["block_idx"].isin([i])].reset_index(drop=True)
                solution_data["block_id"].append(init_block_df["block_id"][0])

                new_school_df = df[df["school_idx"].isin([j])].reset_index(drop=True)
                solution_data["new_school_nces"].append(new_school_df["ncessch"][0])
                new_distance = hs.haversine(
                    (
                        init_block_df["block_centroid_lat"][0],
                        init_block_df["block_centroid_long"][0],
                    ),
                    (new_school_df["lat"][0], new_school_df["long"][0]),
                    unit=hs.Unit.MILES,
                )
                solution_data["new_distance_to_school"].append(new_distance)

    solution_df = pd.DataFrame(data=solution_data)
    output_df = pd.merge(df_blocks, solution_df, on="block_id", how="inner")
    output_df = pd.merge(
        output_df, df_schools, left_on="new_school_nces", right_on="ncessch", how="left"
    )

    output_file = output_file.format(
        objective_value,
        time_ellapsed,
        optimal,
        num_iterations,
        num_bb_nodes,
    )

    if s3_output_dir:
        s3_output_dir = s3_output_dir.format(
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

        output_df.to_csv(
            s3_output_dir + output_file,
            index=False,
        )

    else:
        output_dir = output_dir.format(
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

        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        output_df.to_csv(
            output_dir + output_file,
            index=False,
        )

    return output_file


def print_school_seg_info(x, df, solver, is_cp):
    block_idx = set(df["block_idx"].tolist())
    school_idx = set(df["school_idx"].tolist())
    for j in school_idx:
        school_df = df[df["school_idx"].isin([j])].reset_index(drop=True).iloc[0]
        blocks = []
        for i in block_idx:
            condition_val = False
            if is_cp:
                condition_val = solver.BooleanValue(x[i][j])
            else:
                condition_val = x[i][j].solution_value() > 0.5
            if condition_val:
                blocks.append(i)
        block_df = df[df["block_idx"].isin(blocks)].reset_index(drop=True)
        seg_vals = []
        mapping = {
            "black": "perblk",
            "white": "perwht",
            "asian": "perasn",
            "native": "pernam",
            "hispanic": "perhsp",
            "ell": "perell",
            "frl": "perfrl",
        }
        for r in mapping:
            num = block_df["num_{}_to_school".format(r)].sum()
            denom = block_df["num_total_to_school".format(r)].sum()
            curr = num / denom

            dist_prop = float(school_df["district_{}".format(mapping[r])])
            new_seg = (curr - dist_prop) / (1 - dist_prop)
            # seg_vals.append((new_seg, num, denom))
            seg_vals.append(new_seg)

        print(
            "School {}\n\tblack seg new {} - old {}\n\twhite seg new {} - old {}\n\tnative seg new {} - old {}\n\tasian seg new {} - old {}\n\thispanic seg new {} - old {}\n\tell seg new {} - old {}\n\tfrl seg new {} - old {}\n".format(
                school_df["ncessch"],
                seg_vals[0],
                school_df["seg_index_perblk"],
                seg_vals[1],
                school_df["seg_index_perwht"],
                seg_vals[2],
                school_df["seg_index_perasn"],
                seg_vals[3],
                school_df["seg_index_pernam"],
                seg_vals[4],
                school_df["seg_index_perhsp"],
                seg_vals[5],
                school_df["seg_index_perell"],
                seg_vals[6],
                school_df["seg_index_perfrl"],
            )
        )


def print_dissimilarity_indices(df, objective_value, CATS_TO_INCLUDE):
    df_g = (
        df.groupby("ncessch")
        .agg(
            {
                "num_black_to_school": "sum",
                "num_white_to_school": "sum",
                "num_asian_to_school": "sum",
                "num_native_to_school": "sum",
                "num_hispanic_to_school": "sum",
                "num_ell_to_school": "sum",
                "num_frl_to_school": "sum",
                "num_total_to_school": "sum",
                "total_enrollment": "first",
            }
        )
        .reset_index()
    )
    for cats in CATS_TO_INCLUDE:
        dissim_vals = []
        for key in cats:
            for i in range(0, len(df_g)):
                dissim_vals.append(
                    np.abs(
                        (df_g[key][i] / df_g[key].sum())
                        - (
                            (df_g["num_total_to_school"][i] - df_g[key][i])
                            / (df_g["num_total_to_school"].sum() - df_g[key].sum())
                        )
                    )
                )

        print(
            "{} Dissim new: {} ---- orig: {}".format(
                cats, objective_value, 0.5 * np.sum(dissim_vals)
            )
        )


def print_normalized_exposure_indices(df, objective_value, CATS_TO_INCLUDE):

    df_g = (
        df.groupby("ncessch")
        .agg(
            {
                "num_black_to_school": "sum",
                "num_white_to_school": "sum",
                "num_asian_to_school": "sum",
                "num_native_to_school": "sum",
                "num_hispanic_to_school": "sum",
                "num_ell_to_school": "sum",
                "num_frl_to_school": "sum",
                "num_total_to_school": "sum",
                "total_enrollment": "first",
            }
        )
        .reset_index()
    )
    for cats in CATS_TO_INCLUDE:
        normalized_exposure_vals = []
        for key in cats:
            for i in range(0, len(df_g)):
                normalized_exposure_vals.append(
                    (df_g[key][i] / df_g[key].sum())
                    * (df_g[key][i] / df_g["num_total_to_school"][i])
                )

        P = df_g[key].sum() / df_g["num_total_to_school"].sum()
        objective_index = (objective_value - P) / (1 - P)
        print(
            "{} Normalized exposure new: {} ---- orig: {}".format(
                cats,
                objective_index,
                ((np.sum(normalized_exposure_vals) - P) / (1 - P)),
            )
        )


def print_gini_indices(df, objective_value, CATS_TO_INCLUDE):

    df_g = (
        df.groupby("ncessch")
        .agg(
            {
                "num_black_to_school": "sum",
                "num_white_to_school": "sum",
                "num_asian_to_school": "sum",
                "num_native_to_school": "sum",
                "num_hispanic_to_school": "sum",
                "num_ell_to_school": "sum",
                "num_frl_to_school": "sum",
                "num_total_to_school": "sum",
                "total_enrollment": "first",
            }
        )
        .reset_index()
    )
    for cats in CATS_TO_INCLUDE:
        gini_vals = []
        for key in cats:
            for i in range(0, len(df_g)):
                for j in range(0, len(df_g)):
                    gini_vals.append(
                        (
                            df_g["num_total_to_school"][i]
                            * df_g["num_total_to_school"][j]
                        )
                        * np.abs(
                            (df_g[key][i] / df_g["num_total_to_school"][i])
                            - (df_g[key][j] / df_g["num_total_to_school"][j])
                        )
                    )

        P = df_g[key].sum() / df_g["num_total_to_school"].sum()
        denom = 2 * (df_g["num_total_to_school"].sum() ** 2) * P * (1 - P)
        objective_index = objective_value / denom
        print(
            "{} Gini new: {} ---- orig: {}".format(
                cats, objective_index, (np.sum(gini_vals) / denom)
            )
        )


def distill_network_info_for_blocks(df, blocks_networks):
    all_neighbors_assigned_to_current_school = [[] for i in range(0, len(df))]

    for s in blocks_networks:
        blocks_zoned_to_school = df[df["school_idx"].isin([int(s)])]
        g = blocks_networks[s]
        all_nodes = g.nodes(data=True)
        for _, b in blocks_zoned_to_school.iterrows():

            # If the block is not in the graph, skip it
            if not b["block_idx"] in g:
                continue

            # Get only the neighbors that are also zoned for the same school as the block
            neighbors = g.neighbors(b["block_idx"])
            all_neighbors_assigned_to_current_school[b["block_idx"]] = list(
                filter(
                    lambda x: all_nodes[x]["attrs"]["orig_school_idx"] == int(s),
                    neighbors,
                )
            )

    return all_neighbors_assigned_to_current_school


def bfs_build_network_iterative(root_block, df_shapes, g):

    root_node = (
        int(root_block.block_idx),
        {"orig_school_idx": int(root_block.school_idx), "dist_from_root": 0},
        root_block,
    )
    g.add_node(
        root_node[0],
        attrs=root_node[1],
    )
    nodes_to_explore = [root_node]
    while len(nodes_to_explore) > 0:
        curr_node = nodes_to_explore.pop(0)
        neighbors = df_shapes[~df_shapes.geometry.disjoint(curr_node[2].geometry)]

        for index, n in neighbors.iterrows():
            new_node = (
                int(n.block_idx),
                {
                    "dist_from_root": curr_node[1]["dist_from_root"] + 1,
                    "orig_school_idx": int(n.school_idx),
                },
                n,
            )
            if not (new_node[0] in g):
                g.add_node(new_node[0], attrs=new_node[1])
                nodes_to_explore.append(new_node)

            g.add_edge(curr_node[0], new_node[0])


def identify_island_nodes(g, curr_school_idx):
    all_nodes = g.nodes(data=True)
    all_nodes_no_data = g.nodes()
    root_node = list(
        filter(
            lambda x: all_nodes[x]["attrs"]["dist_from_root"] == 0, all_nodes_no_data
        )
    )[0]
    nodes_assigned_to_school = list(
        filter(
            lambda x: all_nodes[x]["attrs"]["orig_school_idx"] == curr_school_idx,
            all_nodes_no_data,
        )
    )

    g_sub = g.subgraph(nodes_assigned_to_school)
    island_attrs = {}
    for n in all_nodes_no_data:
        if not n in nodes_assigned_to_school:
            island_attrs[n] = ""
        else:
            if not root_node in g_sub:
                island_attrs[n] = 1
            elif not nx.has_path(g_sub, n, root_node):
                island_attrs[n] = 1
            else:
                island_attrs[n] = 0

    nx.set_node_attributes(g, island_attrs, "is_island_wrt_orig_school")


def load_and_prep_data(
    district_id,
    year,
    state,
    df,
    travel_key="times",
    solver_files_root="models/solver_files/",
    solver_data_file="{}{}/{}/prepped_file_for_solver_{}.csv",
    travel_distances_matrix_file="{}{}/{}/prepped_distance_matrix_{}.csv",
    travel_times_matrix_file="{}{}/{}/prepped_travel_time_matrix_{}.csv",
    blocks_networks_file="{}{}/{}/blocks_networks_{}.json",
    blocks_shape_file="data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp",
    state_codes_file="data/state_codes.csv",
):

    print(district_id)

    if not os.path.exists(solver_files_root + year + state + "/"):
        Path(solver_files_root + year + "/" + state + "/").mkdir(
            parents=True, exist_ok=True
        )

    # Create index values for block IDs and schools
    df["block_idx"] = pd.Categorical(
        df["block_id"], categories=df["block_id"].unique()
    ).codes
    df["school_idx"] = pd.Categorical(
        df["ncessch"], categories=df["ncessch"].unique()
    ).codes

    df.to_csv(
        solver_data_file.format(solver_files_root, year, state, district_id),
        index=False,
    )

    block_idx = set(df["block_idx"].tolist())
    school_idx = set(df["school_idx"].tolist())
    travel_matrix = [[0 for j in school_idx] for i in block_idx]
    blocks_networks = {}
    df_school = df.groupby("school_idx").agg(
        {"school_idx": "first", "lat": "first", "long": "first"}
    )

    # If there are more than 200 elementary schools in the district,
    # break out and return due to computational resource limitations
    if len(df_school) > 200:
        return

    if travel_key == "distances":
        travel_function = get_distance_for_coord_pair
        travel_matrix_file = travel_distances_matrix_file.format(
            solver_files_root, year, state, district_id
        )
        first_block_coord = "block_centroid_lat"
        second_block_coord = "block_centroid_long"
        first_school_coord = "lat"
        second_school_coord = "long"
    else:
        travel_function = get_travel_time_for_coord_pair
        travel_matrix_file = travel_times_matrix_file.format(
            solver_files_root, year, state, district_id
        )
        first_block_coord = "block_centroid_long"
        second_block_coord = "block_centroid_lat"
        first_school_coord = "long"
        second_school_coord = "lat"

    # If selected travel matrix exists, load it; otherwise, compute it
    if os.path.exists(
        travel_matrix_file.format(solver_files_root, year, state, district_id)
    ):
        travel_matrix = read_dict(
            travel_matrix_file.format(solver_files_root, year, state, district_id)
        )

    else:
        for i in range(0, len(df)):
            # print(i / len(df))
            block_coords = (
                df[first_block_coord][i],
                df[second_block_coord][i],
            )
            for j in range(0, len(df_school)):
                school_coords = (
                    df_school[first_school_coord][j],
                    df_school[second_school_coord][j],
                )
                coords = (block_coords, school_coords)
                travel_matrix[df["block_idx"][i]][
                    df_school["school_idx"][j]
                ] = travel_function(coords)

        if len(travel_matrix) > 0:
            print("Num total: ", len(travel_matrix) * len(travel_matrix[0]))

        travel_matrix = np.array(travel_matrix, dtype=float).tolist()
        write_dict(
            travel_matrix_file.format(solver_files_root, year, state, district_id),
            travel_matrix,
        )

    # If blocks networks file exists, load it; otherwise, compute it
    if os.path.exists(
        blocks_networks_file.format(solver_files_root, year, state, district_id)
    ):
        blocks_networks = read_dict(
            blocks_networks_file.format(solver_files_root, year, state, district_id)
        )
        for s in blocks_networks:
            blocks_networks[s] = json_graph.node_link_graph(blocks_networks[s])
    else:
        df_states = pd.read_csv(state_codes_file, dtype=str)
        df_states = df_states[df_states["abbrev"].isin([state])].reset_index()
        state_fips = df_states["fips_code"][0]
        print("\tLoading blocks data ...")
        blocks = gpd.read_file(blocks_shape_file.format(state_fips, state_fips))
        blocks["GEOID20"] = blocks["GEOID20"].astype(int)
        df_shapes = gpd.GeoDataFrame(
            pd.merge(df, blocks, left_on="block_id", right_on="GEOID20", how="inner")
        )

        for s in school_idx:
            # Get the block that contains the school
            print("\tBuilding graph for school: ", s)
            df_school = df[df["school_idx"].isin([s])].iloc[0]
            school_lat_long = Point(df_school["long"], df_school["lat"])
            school_check = df_shapes.contains(school_lat_long, align=True)
            df_shapes["contains_curr_school"] = school_check

            # If there's no block containing a given school,
            # set the block closest to the school's location and zoned
            # to that school as the containing block
            try:
                block_containing_school = df_shapes[
                    df_shapes["contains_curr_school"] == True
                ].iloc[0]
            except Exception as e:
                print(e)
                block_containing_school = (
                    df_shapes[df_shapes["school_idx"].isin([s])]
                    .sort_values(by="distance_to_school", ascending=True)
                    .iloc[0]
                )

            g = nx.Graph()
            bfs_build_network_iterative(block_containing_school, df_shapes, g)
            g.remove_edges_from(nx.selfloop_edges(g))
            print("\t", len(g.nodes()), len(g.edges()))
            print("\tAdding island attributes ...")
            identify_island_nodes(g, s)
            blocks_networks[str(s)] = json_graph.node_link_data(g)
            # break

        write_dict(
            blocks_networks_file.format(solver_files_root, year, state, district_id),
            blocks_networks,
        )

        # Turn into networkx graphs before returning
        for s in blocks_networks:
            blocks_networks[s] = json_graph.node_link_graph(blocks_networks[s])

    # Identify neighbors at current school
    df["neighbors_at_current_school"] = distill_network_info_for_blocks(
        df, blocks_networks
    )
    return df, travel_matrix, blocks_networks


def compute_solver_files_parallel(
    data_root_folder="data/derived_data/{}/",
    input_file_schools="{}{}/schools_file_for_assignment.csv",
    input_file_blocks="{}{}/blocks_file_for_assignment.csv",
    solver_file_root_folder="models/solver_files/{}/{}/",
    year="2122",
    # states_to_process=["NC"],
    processing_function=load_and_prep_data,
):

    N_THREADS = 10
    data_root_folder = data_root_folder.format(year)
    all_districts_to_process = []
    states_to_process = os.listdir(data_root_folder)
    for state in states_to_process:

        if not os.path.exists(solver_file_root_folder.format(year, state)):
            Path(solver_file_root_folder.format(year, state)).mkdir(
                parents=True, exist_ok=True
            )

        # districts_already_processed = set(
        #     [
        #         int(f.split("_")[-1].split(".csv")[0])
        #         for f in os.listdir("models/solver_files/{}/{}/".format(year, state))
        #         if f.startswith("prepped_travel_time_matrix")
        #     ]
        # )

        df_b = pd.read_csv(input_file_blocks.format(data_root_folder, state))
        df_s = pd.read_csv(input_file_schools.format(data_root_folder, state))
        df = pd.merge(df_b, df_s, on="ncessch", how="left")
        # district_ids_to_process = (
        #     set(df["leaid"].tolist()) - districts_already_processed
        # )
        district_ids_to_process = set(df["leaid"].tolist())
        districts_to_process = []
        for d in district_ids_to_process:
            districts_to_process.append(
                (d, year, state, df[df["leaid"].isin([int(d)])].reset_index())
            )
        all_districts_to_process.extend(districts_to_process)

    print("Starting parallel processing ...")
    print(len(all_districts_to_process))
    from multiprocessing import Pool

    p = Pool(N_THREADS)
    p.starmap(processing_function, all_districts_to_process)

    p.terminate()
    p.join()


def update_names_for_fips_starting_with_zero(input_dir="models/solver_files/2122/"):
    for state in os.listdir(input_dir):
        for f in os.listdir(os.path.join(input_dir, state)):
            parts = f.split("_")
            district_id = parts[-1].split(".")[0]
            ext = parts[-1].split(".")[1]
            parts.pop()
            if len(district_id) < 7:
                district_id = "0" + district_id
                parts.append(district_id)
                new_name = "_".join(parts) + ".{}".format(ext)
                os.rename(
                    os.path.join(input_dir, state, f),
                    os.path.join(input_dir, state, new_name),
                )


def delete_files_that_should_have_fips_start_with_zero(
    input_dir="models/solver_files/2122/",
):
    for state in os.listdir(input_dir):
        print(state)
        for f in os.listdir(os.path.join(input_dir, state)):
            parts = f.split("_")
            district_id = parts[-1].split(".")[0]
            ext = parts[-1].split(".")[1]
            parts.pop()
            if len(district_id) < 7:
                district_id = "0" + district_id
                parts.append(district_id)
                new_name = "_".join(parts) + ".{}".format(ext)
                os.remove(
                    os.path.join(input_dir, state, f),
                )


if __name__ == "__main__":
    # print("Loading data ...")
    # # district_id = "5102100"
    # district_id = "5101800"
    # # district_id = "5101290"
    # year = "1516"
    # state = "VA"
    # input_file_schools = "data/derived_data/{}/{}/schools_file_for_assignment.csv"
    # input_file_blocks = "data/derived_data/{}/{}/blocks_file_for_assignment.csv"
    # df_s = pd.read_csv(input_file_schools.format(year, state))
    # df_b = pd.read_csv(input_file_blocks.format(year, state))
    # df = pd.merge(df_b, df_s, on="ncessch", how="left")
    # df = df[df["leaid"].isin([int(district_id)])].reset_index(drop=True)
    # load_and_prep_data(district_id, year, state, df)

    # compute_solver_files_parallel()
    # update_names_for_fips_starting_with_zero()
    delete_files_that_should_have_fips_start_with_zero()
