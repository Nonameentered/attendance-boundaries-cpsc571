from ortools.sat.python import cp_model

from analyze_model_solution import identify_switched_blocks
from constants import *
from objective_functions import min_total_segregation, min_max_segregation
from utils.header import *
from utils.model_utils import (
    output_solver_solution,
    print_dissimilarity_indices,
    print_normalized_exposure_indices,
    print_gini_indices,
    load_and_prep_data,
)


def get_neighbors_closer_to_school(g, all_nodes, b):
    neighbors = g.neighbors(b)
    closer_blocks = list(
        filter(
            lambda x: all_nodes[x]["attrs"]["dist_from_root"]
            < all_nodes[b]["attrs"]["dist_from_root"],
            neighbors,
        )
    )
    return closer_blocks


def create_variables(model, df):
    # Assignment of blocks to schools
    block_idx = set(df["block_idx"].tolist())
    school_idx = set(df["school_idx"].tolist())
    x = [
        [model.NewBoolVar("{},{}".format(i, j)) for j in school_idx] for i in block_idx
    ]

    # Add in hints
    for i in block_idx:
        curr_block = df[df["block_idx"].isin([i])].iloc[0]
        assigned_school_j = int(curr_block["school_idx"])
        for j in school_idx:
            if j == assigned_school_j:
                model.AddHint(x[i][j], True)
            else:
                model.AddHint(x[i][j], False)

    return x


def set_constraints(
    model,
    x,
    df,
    travel_matrix,
    blocks_networks,
    max_percent_travel_increase,
    max_percent_size_increase,
    percent_neighbors_rezoned_together,
    enforce_contiguity,
):
    ########## Pre-process data ##########
    block_idx = set(df["block_idx"].tolist())
    school_idx = set(df["school_idx"].tolist())

    school_populations = defaultdict(Counter)
    block_populations = defaultdict(Counter)

    for i in range(0, len(df)):

        # Save the current total student population of the block estimated as sent to the zoned school
        for cats in ALL_CATS.values():
            for key in cats:
                block_populations[df["block_idx"][i]][key] = int(df[key][i])

    # Save total enrollment at school
    for j in school_idx:
        curr_df = df[df["school_idx"].isin([j])].reset_index(drop=True)
        school_populations[j]["total_enrollment"] = int(curr_df["total_enrollment"][0])
        for cats in ALL_CATS.values():
            for key in cats:
                val = cats[key]
                if val == "pertotal":
                    continue
                school_populations[j][val] = int(
                    curr_df[val][0] * school_populations[j]["total_enrollment"]
                )

    ########## Add constraints ##########
    for i in block_idx:

        curr_block = df[df["block_idx"].isin([i])].iloc[0]
        assigned_school_j = int(curr_block["school_idx"])

        # ***CONSTRAINT: Each block is assigned to exactly 1 school
        model.Add(sum(x[i][j] for j in school_idx) == 1)

        # **CONSTRAINT: If an element of the travel matrix is NaN, do not allow the row (block)
        # to be assigned to the corresponding column (school) UNLESS it is the block's
        # currently-assigned school
        eligible_schools_for_assignment = set()
        school_idx_except_assigned = set(school_idx)
        school_idx_except_assigned.remove(assigned_school_j)
        for j in school_idx_except_assigned:
            if np.isnan(travel_matrix[i][j]):
                model.Add(x[i][j] == False)
            else:
                eligible_schools_for_assignment.add(j)

        # ***CONSTRAINT: If the current travel time from the block to its zoned school is NaN, do not allow it to be re-assigned
        # Why?  We don't have a sense of what other permissible assignments could be (given we don't know baseline travel time)
        if np.isnan(travel_matrix[i][assigned_school_j]):
            model.Add(x[i][assigned_school_j] == True)

        else:
            # ***CONSTRAINT: Change in travel for each block must be <= max_percent_travel_increase
            max_allowed_travel_time = int(
                SCALING[0]
                * np.round(
                    (1 + max_percent_travel_increase)
                    * travel_matrix[i][assigned_school_j],
                    decimals=SCALING[1],
                )
            )
            for j in eligible_schools_for_assignment:
                travel_to_this_school = int(
                    SCALING[0]
                    * np.round(
                        travel_matrix[i][j],
                        decimals=SCALING[1],
                    )
                )

                if travel_to_this_school > max_allowed_travel_time:
                    model.Add(x[i][j] == False)

        # ***CONSTRAINT: Require that x% of the block's current neighbors are rezoned with the block
        neighbors_at_current_school = curr_block["neighbors_at_current_school"]
        threshold = min(
            int(
                np.ceil(
                    percent_neighbors_rezoned_together
                    * len(neighbors_at_current_school)
                )
            ),
            len(neighbors_at_current_school),
        )
        for j in school_idx:
            model.Add(
                sum(x[n][j] for n in neighbors_at_current_school) >= threshold
            ).OnlyEnforceIf(x[i][j])

    for j in school_idx:
        # ***CONSTRAINT: Change in number of students sent from all blocks to school must be <= max_percent_size_increase
        model.Add(
            SCALING[0]
            * sum(
                x[i][j] * block_populations[i]["num_total_to_school"] for i in block_idx
            )
            <= int(
                SCALING[0]
                * np.round(
                    (1 + max_percent_size_increase)
                    * school_populations[j]["total_enrollment"],
                    decimals=SCALING[1],
                )
            )
        )

        # ***CONSTRAINT: Only contiguous blocks can be zoned to the same school
        if enforce_contiguity:
            curr_graph = blocks_networks[str(j)]
            all_nodes = curr_graph.nodes(data=True)
            all_nodes_no_data = curr_graph.nodes()
            root_block_id = [
                x for x, y in all_nodes if y["attrs"]["dist_from_root"] == 0
            ][0]

            # Iterate over all of the blocks except for the root and those that are already islands
            # to set contiguity constraints
            already_islands = list(
                filter(
                    lambda x: all_nodes[x]["is_island_wrt_orig_school"] == 1,
                    all_nodes_no_data,
                )
            )
            block_idx_to_enforce_contiguity_for = list(
                (
                    set(block_idx) - set([root_block_id]) - set(already_islands)
                ).intersection(all_nodes_no_data)
            )

            for b in block_idx_to_enforce_contiguity_for:
                neighbors_closer_than_b_to_j = set(
                    get_neighbors_closer_to_school(curr_graph, all_nodes, b)
                )

                if all_nodes[b]["attrs"]["orig_school_idx"] == j:
                    curr_block = df[df["block_idx"].isin([b])].iloc[0]
                    curr_block_neighbors = set(
                        curr_block["neighbors_at_current_school"]
                    )
                    curr_len = len(
                        curr_block_neighbors.intersection(neighbors_closer_than_b_to_j)
                    )

                    if curr_len == 0:
                        # model.Add(x[b][j] == 1)
                        # continue

                        nodes_assigned_to_school = list(
                            filter(
                                lambda x: (
                                    all_nodes[x]["attrs"]["orig_school_idx"] == j
                                    and all_nodes[x]["is_island_wrt_orig_school"] != 1
                                ),
                                all_nodes_no_data,
                            )
                        )

                        g_sub = curr_graph.subgraph(nodes_assigned_to_school)
                        for n in nodes_assigned_to_school:
                            try:
                                sp_length = nx.shortest_path_length(
                                    g_sub, source=n, target=root_block_id
                                )
                                all_nodes[n]["attrs"]["dist_from_root"] = sp_length
                            except Exception as e:
                                continue

                        neighbors_closer_than_b_to_j = set(
                            get_neighbors_closer_to_school(g_sub, all_nodes, b)
                        )

                # print(j, b, neighbors_closer_than_b_to_j)
                # block b is assigned to school j only if at least one (not necessarily all)
                # of its neighbors n that is closer to j is also assigned to j. Does
                # not apply to blocks that were previously islands — they can still be
                # islands
                # if all_nodes[b]["attrs"]["orig_school_idx"] != j:
                model.Add(
                    sum(x[n][j] for n in neighbors_closer_than_b_to_j) > 0
                ).OnlyEnforceIf(x[b][j])


def preprocess_objective_data(df):
    block_idx = set(df["block_idx"].tolist())
    school_idx = set(df["school_idx"].tolist())
    block_populations = defaultdict(Counter)
    dist_pop_per_cat = Counter()

    for i in range(0, len(df)):
        for cats in ALL_CATS.values():
            for key in cats:
                block_populations[df["block_idx"][i]][key] = int(df[key][i])
                dist_pop_per_cat[key] += int(df[key][i])

    return block_idx, school_idx, block_populations, dist_pop_per_cat


def set_objective_function_dissimilarity(
    model, x, df, CATS_TO_INCLUDE, objective_function
):
    ########## Pre-process data ##########
    (
        block_idx,
        school_idx,
        block_populations,
        dist_pop_per_cat,
    ) = preprocess_objective_data(df)

    ########## Set objective function ##########

    objective_terms = []
    for j in school_idx:
        for cats in CATS_TO_INCLUDE:
            for key in cats:

                school_cat_terms = []
                school_total_terms = []
                for i in block_idx:
                    school_total_terms.append(
                        x[i][j] * block_populations[i]["num_total_to_school"]
                    )
                    school_cat_terms.append(x[i][j] * block_populations[i][key])

                # Scaling and prepping to apply division equality constraint,
                # required to do divisions in CP-SAT
                scaled_total_cat_students_at_school = model.NewIntVar(
                    0, SCALING[0] ** 2, ""
                )

                model.Add(
                    scaled_total_cat_students_at_school
                    == SCALING[0] * sum(school_cat_terms)
                )

                scaled_total_non_cat_students_at_school = model.NewIntVar(
                    0, SCALING[0] ** 2, ""
                )
                model.Add(
                    scaled_total_non_cat_students_at_school
                    == SCALING[0] * (sum(school_total_terms) - sum(school_cat_terms))
                )

                # Fraction of cat students that are at this school
                cat_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
                model.AddDivisionEquality(
                    cat_ratio_at_school,
                    scaled_total_cat_students_at_school,
                    dist_pop_per_cat[key],
                )

                # Fraction of non-cat students that are at this school
                non_cat_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
                model.AddDivisionEquality(
                    non_cat_ratio_at_school,
                    scaled_total_non_cat_students_at_school,
                    dist_pop_per_cat["num_total_to_school"] - dist_pop_per_cat[key],
                )

                # Computing dissimilarity index
                diff_val = model.NewIntVar(-(SCALING[0] ** 2), SCALING[0] ** 2, "")
                model.Add(diff_val == cat_ratio_at_school - non_cat_ratio_at_school)
                obj_term_to_add = model.NewIntVar(0, SCALING[0] ** 2, "")
                model.AddAbsEquality(
                    obj_term_to_add,
                    diff_val,
                )

                objective_terms.append(obj_term_to_add)

    if objective_function == "min_total_segregation":
        min_total_segregation(model, objective_terms)
    elif objective_function == "min_max_segregation":
        min_max_segregation(model, objective_terms)


def set_objective_function_normalized_exposure(
    model, x, df, CATS_TO_INCLUDE, objective_function
):
    ########## Pre-process data ##########
    (
        block_idx,
        school_idx,
        block_populations,
        dist_pop_per_cat,
    ) = preprocess_objective_data(df)

    ########## Set objective function ##########

    objective_terms = []
    for j in school_idx:
        for cats in CATS_TO_INCLUDE:
            for key in cats:

                school_cat_terms = []
                school_total_terms = []
                for i in block_idx:
                    school_total_terms.append(
                        x[i][j] * block_populations[i]["num_total_to_school"]
                    )
                    school_cat_terms.append(x[i][j] * block_populations[i][key])

                total_students_at_school = model.NewIntVar(1, MAX_TOTAL_STUDENTS, "")
                model.Add(total_students_at_school == sum(school_total_terms))

                # Scaling and prepping to apply division equality constraint,
                # required to do divisions in CP-SAT
                scaled_total_cat_students_at_school = model.NewIntVar(
                    0, SCALING[0] ** 2, ""
                )

                model.Add(
                    scaled_total_cat_students_at_school
                    == SCALING[0] * sum(school_cat_terms)
                )

                # Fraction of cat students across district that are at this school
                dist_cat_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
                model.AddDivisionEquality(
                    dist_cat_ratio_at_school,
                    scaled_total_cat_students_at_school,
                    dist_pop_per_cat[key],
                )

                # Fraction of students at school who belong to cat
                school_cat_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
                model.AddDivisionEquality(
                    school_cat_ratio_at_school,
                    scaled_total_cat_students_at_school,
                    total_students_at_school,
                )

                # Computing term for normalized exposure index
                obj_term_to_add = model.NewIntVar(0, SCALING[0] ** 2, "")
                model.AddMultiplicationEquality(
                    obj_term_to_add,
                    [dist_cat_ratio_at_school, school_cat_ratio_at_school],
                )

                objective_terms.append(obj_term_to_add)

    model.Minimize(sum(objective_terms))


def set_objective_function_gini(model, x, df, CATS_TO_INCLUDE, objective_function):
    ########## Pre-process data ##########
    (
        block_idx,
        school_idx,
        block_populations,
        dist_pop_per_cat,
    ) = preprocess_objective_data(df)

    ########## Set objective function ##########

    objective_terms = []

    for cats in CATS_TO_INCLUDE:
        for key in cats:
            for j in school_idx:
                school_j_cat_terms = []
                school_j_total_terms = []
                for i in block_idx:
                    school_j_total_terms.append(
                        x[i][j] * block_populations[i]["num_total_to_school"]
                    )
                    school_j_cat_terms.append(x[i][j] * block_populations[i][key])

                total_students_at_school_j = model.NewIntVar(1, MAX_TOTAL_STUDENTS, "")
                model.Add(total_students_at_school_j == sum(school_j_total_terms))

                # Scaling and prepping to apply division equality constraint,
                # required to do divisions in CP-SAT
                scaled_total_cat_students_at_school_j = model.NewIntVar(
                    0, SCALING[0] * MAX_STUDENTS_PER_CAT, ""
                )

                model.Add(
                    scaled_total_cat_students_at_school_j
                    == SCALING[0] * sum(school_j_cat_terms)
                )

                # Fraction of students at school who belong to cat
                school_j_cat_ratio_at_school = model.NewIntVar(
                    0, SCALING[0] * MAX_STUDENTS_PER_CAT, ""
                )
                model.AddDivisionEquality(
                    school_j_cat_ratio_at_school,
                    scaled_total_cat_students_at_school_j,
                    total_students_at_school_j,
                )

                for k in school_idx:
                    # print(j, k)
                    school_k_cat_terms = []
                    school_k_total_terms = []
                    for i in block_idx:
                        school_k_total_terms.append(
                            x[i][k] * block_populations[i]["num_total_to_school"]
                        )
                        school_k_cat_terms.append(x[i][k] * block_populations[i][key])

                    total_students_at_school_k = model.NewIntVar(
                        1, MAX_TOTAL_STUDENTS, ""
                    )
                    model.Add(total_students_at_school_k == sum(school_k_total_terms))

                    # Scaling and prepping to apply division equality constraint,
                    # required to do divisions in CP-SAT
                    scaled_total_cat_students_at_school_k = model.NewIntVar(
                        0, SCALING[0] * MAX_STUDENTS_PER_CAT, ""
                    )

                    model.Add(
                        scaled_total_cat_students_at_school_k
                        == SCALING[0] * sum(school_k_cat_terms)
                    )

                    # Fraction of students at school who belong to cat
                    school_k_cat_ratio_at_school = model.NewIntVar(
                        0, SCALING[0] * MAX_STUDENTS_PER_CAT, ""
                    )
                    model.AddDivisionEquality(
                        school_k_cat_ratio_at_school,
                        scaled_total_cat_students_at_school_k,
                        total_students_at_school_k,
                    )

                    # Computing terms for gini index
                    school_pair_pop_prod = model.NewIntVar(1, MAX_TOTAL_STUDENTS**2, "")
                    model.AddMultiplicationEquality(
                        school_pair_pop_prod,
                        [total_students_at_school_j, total_students_at_school_k],
                    )
                    diff_val = model.NewIntVar(
                        -(SCALING[0] * MAX_STUDENTS_PER_CAT),
                        (SCALING[0] * MAX_STUDENTS_PER_CAT),
                        "",
                    )
                    model.Add(
                        diff_val
                        == school_j_cat_ratio_at_school - school_k_cat_ratio_at_school
                    )
                    ratio_diff_abs_value = model.NewIntVar(0, SCALING[0], "")
                    model.AddAbsEquality(ratio_diff_abs_value, diff_val)
                    obj_term_to_add = model.NewIntVar(
                        0, SCALING[0] * (MAX_TOTAL_STUDENTS**2), ""
                    )
                    model.AddMultiplicationEquality(
                        obj_term_to_add,
                        [school_pair_pop_prod, ratio_diff_abs_value],
                    )

                    objective_terms.append(obj_term_to_add)

    model.Minimize(sum(objective_terms))


def solve_and_output_results(
    optimizing_for="white",
    district_id="2400720",
    year="2122",
    state="MD",
    max_percent_travel_increase=0.5,
    max_percent_size_increase=0.15,
    percent_neighbors_rezoned_together=0.5,
    enforce_contiguity=True,
    objective_function="min_total_segregation",
    objective_type="dissimilarity",
    input_file_schools="data/derived_data/{}/{}/schools_file_for_assignment.csv",
    input_file_blocks="data/derived_data/{}/{}/blocks_file_for_assignment.csv",
):

    # Create the cp model
    model = cp_model.CpModel()

    print("Loading data ...")
    df_s = pd.read_csv(input_file_schools.format(year, state))
    df_b = pd.read_csv(input_file_blocks.format(year, state))
    df = pd.merge(df_b, df_s, on="ncessch", how="left")
    df = df[df["leaid"].isin([int(district_id)])].reset_index(drop=True)
    df, travel_matrix, blocks_networks = load_and_prep_data(
        district_id, year, state, df
    )

    print("Creating variables ...")
    x = create_variables(model, df)

    print("Setting constraints ...")
    # Make sure enforce_contiguity is a bool, in case a string "True" or "False" was passed in
    from distutils.util import strtobool

    enforce_contiguity = strtobool(str(enforce_contiguity))
    set_constraints(
        model,
        x,
        df,
        travel_matrix,
        blocks_networks,
        max_percent_travel_increase,
        max_percent_size_increase,
        percent_neighbors_rezoned_together,
        enforce_contiguity,
    )

    print("Setting objective function {}...".format(objective_type))
    if objective_type == "dissimilarity":
        set_objective_function_dissimilarity(
            model,
            x,
            df,
            [ALL_CATS[optimizing_for]],
            objective_function,
        )
    elif objective_type == "normalized_exposure":
        set_objective_function_normalized_exposure(
            model,
            x,
            df,
            [ALL_CATS[optimizing_for]],
            objective_function,
        )
    elif objective_type == "gini":
        set_objective_function_gini(
            model,
            x,
            df,
            [ALL_CATS[optimizing_for]],
            objective_function,
        )

    print("Solving ...")
    solver = cp_model.CpSolver()

    # Sets a time limit for solver
    solver.parameters.max_time_in_seconds = MAX_SOLVER_TIME

    # Adding parallelism
    solver.parameters.num_search_workers = NUM_SOLVER_THREADS

    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

        print("Outputting solver solution ...")
        solution_file = output_solver_solution(
            x,
            solver,
            True,
            solver.ObjectiveValue() / SCALING[0],
            solver.WallTime(),
            solver.NumBranches(),
            solver.NumConflicts(),
            status,
            optimizing_for,
            df,
            district_id,
            year,
            state,
            max_percent_travel_increase,
            max_percent_size_increase,
            objective_function,
            percent_neighbors_rezoned_together,
            "True" if enforce_contiguity else "False",
        )

        if objective_type == "dissimilarity":
            print_dissimilarity_indices(
                df,
                (solver.ObjectiveValue() / SCALING[0]) / 2,
                [ALL_CATS[optimizing_for]],
            )
        elif objective_type == "normalized_exposure":
            print_normalized_exposure_indices(
                df,
                (solver.ObjectiveValue() / (SCALING[0] ** 2)),
                [ALL_CATS[optimizing_for]],
            )
        elif objective_type == "gini":
            print_gini_indices(
                df,
                (solver.ObjectiveValue() / SCALING[0]),
                [ALL_CATS[optimizing_for]],
            )

        identify_switched_blocks(
            district_id,
            year,
            state,
            solution_file,
            max_percent_travel_increase,
            max_percent_size_increase,
            optimizing_for,
            objective_function,
            percent_neighbors_rezoned_together,
            "True" if enforce_contiguity else "False",
        )

        # # Output map of rezoning
        # viz_assignments(
        #     district_id,
        #     year,
        #     state,
        #     solution_file,
        #     max_percent_travel_increase,
        #     max_percent_size_increase,
        #     optimizing_for,
        #     objective_function,
        #     percent_neighbors_rezoned_together,
        #     "True" if enforce_contiguity else "False",
        #     pd.DataFrame(),
        # )

    else:
        output_dir = "models/results/{}/{}/{}/{}_{}_{}_{}_{}_{}/".format(
            year,
            state,
            district_id,
            max_percent_travel_increase,
            max_percent_size_increase,
            optimizing_for,
            objective_function,
            percent_neighbors_rezoned_together,
            "True" if enforce_contiguity else "False",
        )

        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        write_dict(output_dir + "solver_output.json", {"status": status})
        print("Status: ", status)
        # print(model.Validate())


if __name__ == "__main__":
    solve_and_output_results()
