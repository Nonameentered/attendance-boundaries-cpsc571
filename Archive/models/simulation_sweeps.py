from utils.header import *
from assignment_cp_sat import solve_and_output_results
from constants import MAX_SOLVER_TIME


def generate_year_state_sweep_configs(
    year="2122",
    states=["MD"],
    max_cluster_node_time=43200,
    total_cluster_tasks_per_group=500,
    districts_to_process=[],
    input_dir="data/derived_data/{}/",
    output_dir="models/sweep_configs/{}_md_expanded/",
):
    max_percent_distance_increases = [0.25, 0.5, 0.75, 1, 1.5, 2]
    max_percent_size_increases = [0.1, 0.15, 0.2]
    percent_neighbors_rezoned_together = [0, 0.5]
    enforce_contiguity = [True, False]
    optimizing_for = ["white"]
    objective_functions = ["min_total_segregation"]

    # TODO: comment out for entire state or country-level sims
    # districts_to_process = pd.read_csv(
    #     "data/school_covariates/top_100_largest_districts.csv", dtype=str
    # )["leaid"].tolist()
    districts_to_process = ["3904475"]

    sweeps = {
        "district_id": [],
        "max_percent_travel_increase": [],
        "max_percent_size_increase": [],
        "percent_neighbors_rezoned_together": [],
        "optimizing_for": [],
        "enforce_contiguity": [],
        "objective_function": [],
        "year": [],
        "state": [],
    }

    for state in os.listdir(input_dir.format(year)):
        # for state in states:
        input_file_schools = os.path.join(
            input_dir.format(year), state, "schools_file_for_assignment.csv"
        )
        df = pd.read_csv(input_file_schools.format(year, state), dtype=str)
        df["num_blocks"] = df["num_blocks"].astype(int)
        df_g = (
            df.groupby(["leaid"], as_index=False)
            .agg({"leaid": "first", "ncessch": "count", "num_blocks": "sum"})
            .rename(columns={"ncessch": "num_schools"})
            .sort_values(by="num_schools", ascending=False)
        )

        # Identify districts that have > 1 school and <= 200 schools
        df_g = df_g[
            (df_g["num_schools"] > 1) & (df_g["num_schools"] <= 200)
        ].reset_index()
        if len(districts_to_process) > 0:
            df_g = df_g[df_g["leaid"].isin(districts_to_process)].reset_index()

        for i in range(0, len(df_g)):
            for d in max_percent_distance_increases:
                for s in max_percent_size_increases:
                    for n in percent_neighbors_rezoned_together:
                        for opt in optimizing_for:
                            for c in enforce_contiguity:
                                for obj in objective_functions:
                                    sweeps["year"].append(year)
                                    sweeps["state"].append(state)
                                    sweeps["district_id"].append(df_g["leaid"][i])
                                    sweeps["max_percent_travel_increase"].append(d)
                                    sweeps["max_percent_size_increase"].append(s)
                                    sweeps["percent_neighbors_rezoned_together"].append(
                                        n
                                    )
                                    sweeps["optimizing_for"].append(opt)
                                    sweeps["objective_function"].append(obj)
                                    sweeps["enforce_contiguity"].append(c)

    # Make directory for simulation sweeps
    if not os.path.exists(output_dir.format(year)):
        Path(output_dir.format(year)).mkdir(parents=True, exist_ok=True)

    # Create groups to run on cluster
    df_out = pd.DataFrame(data=sweeps).sample(frac=1)
    num_jobs_per_group = int(
        np.floor(max_cluster_node_time / MAX_SOLVER_TIME)
        * total_cluster_tasks_per_group
    )
    num_cluster_groups = int(np.ceil(len(df_out) / num_jobs_per_group))
    for i in range(0, num_cluster_groups):
        df_curr = df_out.iloc[(i * num_jobs_per_group) : ((i + 1) * num_jobs_per_group)]
        df_curr.to_csv(output_dir.format(year, state) + str(i) + ".csv", index=False)


def generate_file_to_rerun_failed_sweeps(
    results_file="data/prepped_csvs_for_analysis/simulation_outputs/md_2122_dissim/consolidated_simulation_results.csv",
    sweeps_dirs=[
        "models/sweep_configs/2122_MD_dissim/",
    ],
    output_dir="models/sweep_configs/2122_MD_dissim_rerun/",
    max_cluster_node_time=43200,
    total_cluster_tasks_per_group=500,
):
    df_res = pd.read_csv(results_file)
    df_res["objective_function"] = df_res["objective_function"] + "_segregation"
    df_sim = pd.DataFrame()
    for curr_sweeps_dir in sweeps_dirs:
        for f in os.listdir(curr_sweeps_dir):
            df_curr = pd.read_csv(curr_sweeps_dir + f)
            df_sim = df_sim.append(df_curr)

    df_remaining = pd.merge(
        df_sim,
        df_res,
        left_on=[
            "district_id",
            "max_percent_travel_increase",
            "max_percent_size_increase",
            "percent_neighbors_rezoned_together",
            "optimizing_for",
            "objective_function",
        ],
        right_on=[
            "district_id",
            "travel_time_threshold",
            "school_size_threshold",
            "community_cohesion_threshold",
            "cat_optimizing_for",
            "objective_function",
        ],
        how="left",
    )
    df_remaining = df_remaining[df_remaining["config"].isna()][
        [
            "district_id",
            "max_percent_travel_increase",
            "max_percent_size_increase",
            "percent_neighbors_rezoned_together",
            "optimizing_for",
            "objective_function",
        ]
    ]
    num_jobs_per_group = int(
        (max_cluster_node_time / MAX_SOLVER_TIME) * total_cluster_tasks_per_group
    )
    num_cluster_groups = int(np.ceil(len(df_remaining) / num_jobs_per_group))
    for i in range(0, num_cluster_groups):
        df_curr = df_remaining.iloc[
            (i * num_jobs_per_group) : ((i + 1) * num_jobs_per_group)
        ]
        df_curr.to_csv(output_dir + str(i) + ".csv", index=False)


def run_sweep_for_chunk(
    chunk_ID,
    num_total_chunks,
    group_ID,
    solver_function=solve_and_output_results,
    sweeps_dir="models/sweep_configs/2122_md_expanded/",
):

    df = pd.read_csv(sweeps_dir + str(group_ID) + ".csv", dtype=str)
    df["max_percent_travel_increase"] = df["max_percent_travel_increase"].astype(float)
    df["max_percent_size_increase"] = df["max_percent_size_increase"].astype(float)
    df["percent_neighbors_rezoned_together"] = df[
        "percent_neighbors_rezoned_together"
    ].astype(float)

    configs = []
    for i in range(0, len(df)):
        configs.append(
            {
                "optimizing_for": df["optimizing_for"][i],
                "district_id": df["district_id"][i],
                "year": df["year"][i],
                "state": df["state"][i],
                "enforce_contiguity": df["enforce_contiguity"][i],
                "max_percent_travel_increase": df["max_percent_travel_increase"][i],
                "max_percent_size_increase": df["max_percent_size_increase"][i],
                "percent_neighbors_rezoned_together": df[
                    "percent_neighbors_rezoned_together"
                ][i],
                "objective_function": df["objective_function"][i],
            }
        )

    remainder = len(df) % num_total_chunks
    chunk_size = max(1, int(np.floor(len(df) / num_total_chunks)))
    if remainder > 0 and len(df) > num_total_chunks:
        chunk_size += 1

    configs_to_compute = configs[
        (chunk_ID * chunk_size) : ((chunk_ID + 1) * chunk_size)
    ]

    for curr_config in configs_to_compute:
        try:
            solver_function(**curr_config)
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    # generate_year_state_sweep_configs()
    # generate_file_to_rerun_failed_sweeps()
    if len(sys.argv) < 4:
        print(
            "Error: need to specify <chunk_ID>, <num_chunks>, and <group_id> for cluster run"
        )
        exit()
    run_sweep_for_chunk(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
