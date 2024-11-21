# optimize.py

import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
from ortools.linear_solver import pywraplp
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def load_data(data_path):
    # Load Schools Data
    df_schools = pd.read_csv(f"{data_path}/Schools_High.csv")
    df_schools["geometry"] = df_schools["geom"].apply(wkt.loads)
    gdf_schools = gpd.GeoDataFrame(df_schools, geometry="geometry")

    # Load Census Tracts Data
    df_tracts = pd.read_csv(f"{data_path}/Census2020_Tracts.csv")
    df_tracts["geometry"] = df_tracts["geom"].apply(wkt.loads)
    gdf_tracts = gpd.GeoDataFrame(df_tracts, geometry="geometry")
    df_tracts["Tract"] = df_tracts["Tract"].astype(str)

    # Load Race Data
    df_race = pd.read_csv(f"{data_path}/RaceByCounty.csv")
    df_race.columns = df_race.iloc[0]
    df_race = df_race[1:]
    df_race["Tract"] = df_race["Geography"].apply(lambda x: x[-11:])
    df_race["Tract"] = df_race["Tract"].astype(str)

    df_blocks = pd.merge(df_tracts, df_race, on="Tract", how="left")

    # Wrangle Population and Race Data
    df_blocks["adj_population"] = pd.to_numeric(
        df_blocks["adj_population"], errors="coerce"
    ).fillna(0)
    df_blocks["Population"] = df_blocks["adj_population"].apply(lambda x: max(x, 1e-6))

    race_columns = [
        " !!Total:!!Owner occupied:!!Householder who is White alone",
        " !!Total:!!Owner occupied:!!Householder who is Black or African American alone",
        " !!Total:!!Owner occupied:!!Householder who is Asian alone",
        " !!Total:!!Renter occupied:!!Householder who is White alone",
        " !!Total:!!Renter occupied:!!Householder who is Black or African American alone",
        " !!Total:!!Renter occupied:!!Householder who is Asian alone",
    ]

    for col in race_columns:
        df_blocks[col] = pd.to_numeric(df_blocks[col], errors="coerce").fillna(0)

    df_blocks["White"] = (
        df_blocks[" !!Total:!!Owner occupied:!!Householder who is White alone"]
        + df_blocks[" !!Total:!!Renter occupied:!!Householder who is White alone"]
    )
    df_blocks["Black"] = (
        df_blocks[
            " !!Total:!!Owner occupied:!!Householder who is Black or African American alone"
        ]
        + df_blocks[
            " !!Total:!!Renter occupied:!!Householder who is Black or African American alone"
        ]
    )
    df_blocks["Asian"] = (
        df_blocks[" !!Total:!!Owner occupied:!!Householder who is Asian alone"]
        + df_blocks[" !!Total:!!Renter occupied:!!Householder who is Asian alone"]
    )

    for race in ["White", "Black", "Asian"]:
        df_blocks[race] = df_blocks[race].apply(lambda x: max(x, 1e-6))

    # Calculate Overall Racial Proportions
    total_white = df_blocks["White"].sum()
    total_black = df_blocks["Black"].sum()
    total_asian = df_blocks["Asian"].sum()
    total_race_population = total_white + total_black + total_asian

    if total_race_population == 0:
        total_race_population = 1e-6

    P_k = {
        "White": total_white / total_race_population,
        "Black": total_black / total_race_population,
        "Asian": total_asian / total_race_population,
    }

    # Compute Centroids
    gdf_blocks = gpd.GeoDataFrame(df_blocks, geometry="geometry")
    gdf_blocks["centroid"] = gdf_blocks["geometry"].centroid

    # Compute Distance Matrix
    dist = {}
    max_distance = 0
    for idx_block, block in tqdm(
        gdf_blocks.iterrows(), total=gdf_blocks.shape[0], desc="Calculating distances"
    ):
        block_name = block["Tract"]
        dist[block_name] = {}
        block_centroid = block["centroid"]
        for idx_school, school in gdf_schools.iterrows():
            school_name = school["Name"]
            school_point = school["geometry"]
            distance = block_centroid.distance(school_point)
            dist[block_name][school_name] = distance
            if distance > max_distance:
                max_distance = distance

    BigM = max_distance + 1

    return {
        "gdf_schools": gdf_schools,
        "gdf_blocks": gdf_blocks,
        "P_k": P_k,
        "dist": dist,
        "BigM": BigM,
        "block_names": list(gdf_blocks["Tract"]),
        "school_names": list(gdf_schools["Name"]),
        "df_blocks": df_blocks,
    }


def solve_optimization(
    data,
    lambda_cost,
    lambda_imbalance,
    lambda_compactness,
    solver_time_limit=30000,
    capacity_slack=1.05,
):
    # Unpack data
    gdf_schools = data["gdf_schools"]
    gdf_blocks = data["gdf_blocks"]
    P_k = data["P_k"]
    dist = data["dist"]
    BigM = data["BigM"]
    block_names = data["block_names"]
    school_names = data["school_names"]
    df_blocks = data["df_blocks"]

    # Create optimization model
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Solver not available.")
        return None

    # Set solver parameters for time limit
    solver.SetTimeLimit(solver_time_limit)  # Time limit in milliseconds

    # Decision variables
    x = {}
    for block in block_names:
        for school in school_names:
            x[(block, school)] = solver.IntVar(0, 1, f"x_{block}_{school}")

    # Auxiliary variables for racial imbalance
    d = {}
    for school in school_names:
        for race in ["White", "Black", "Asian"]:
            d[(school, race)] = solver.NumVar(
                0, solver.infinity(), f"d_{school}_{race}"
            )

    # Variables for total assigned populations
    n_j = {}
    n_jk = {}
    for school in school_names:
        n_j[school] = solver.NumVar(0, solver.infinity(), f"n_{school}")
        for race in ["White", "Black", "Asian"]:
            n_jk[(school, race)] = solver.NumVar(
                0, solver.infinity(), f"n_{school}_{race}"
            )

    # Variables for compactness
    M = {}
    for school in school_names:
        M[school] = solver.NumVar(0, solver.infinity(), f"M_{school}")

    # Constraints

    # Assignment Constraint: Each block must be assigned to exactly one school
    for block in block_names:
        solver.Add(solver.Sum([x[(block, school)] for school in school_names]) == 1)

    # Capacity Constraint: Total assigned population should not exceed capacity (with slack)
    total_population = df_blocks["Population"].sum()
    num_schools = len(school_names)
    capacity_per_school = capacity_slack * (total_population / num_schools)

    for school in school_names:
        solver.Add(
            solver.Sum(
                [
                    df_blocks.loc[df_blocks["Tract"] == block, "Population"].values[0]
                    * x[(block, school)]
                    for block in block_names
                ]
            )
            <= capacity_per_school
        )

    # Define n_j and n_jk (Total assigned populations)
    for school in school_names:
        # Total population assigned to school
        solver.Add(
            n_j[school]
            == solver.Sum(
                [
                    df_blocks.loc[df_blocks["Tract"] == block, "Population"].values[0]
                    * x[(block, school)]
                    for block in block_names
                ]
            )
        )
        for race in ["White", "Black", "Asian"]:
            # Total population of race assigned to school
            solver.Add(
                n_jk[(school, race)]
                == solver.Sum(
                    [
                        df_blocks.loc[df_blocks["Tract"] == block, race].values[0]
                        * x[(block, school)]
                        for block in block_names
                    ]
                )
            )
            # Racial Imbalance Constraints: d_{jk} >= |n_{jk} - P_k * n_j|
            solver.Add(
                d[(school, race)] >= n_jk[(school, race)] - P_k[race] * n_j[school]
            )
            solver.Add(
                d[(school, race)] >= -(n_jk[(school, race)] - P_k[race] * n_j[school])
            )
            solver.Add(d[(school, race)] >= 0)

    # Compactness Constraints: M_j >= d_{ij} - (1 - x_{ij}) * BigM
    for school in school_names:
        for block in block_names:
            distance = dist[block][school]
            solver.Add(M[school] >= distance - (1 - x[(block, school)]) * BigM)

    # Compute Family Cost (Total travel distance for students)
    family_cost_terms = []
    for block in block_names:
        block_population = df_blocks.loc[
            df_blocks["Tract"] == block, "Population"
        ].values[0]
        for school in school_names:
            distance_to_school = dist[block][school]
            cost = block_population * distance_to_school * x[(block, school)]
            family_cost_terms.append(cost)

    family_cost = solver.Sum(family_cost_terms)

    # Racial Imbalance Term
    racial_imbalance = solver.Sum(
        [
            d[(school, race)]
            for school in school_names
            for race in ["White", "Black", "Asian"]
        ]
    )

    # Compactness Penalty Term
    compactness_penalty = solver.Sum([M[school] for school in school_names])

    # Objective Function: Minimize weighted sum of family cost, racial imbalance, and compactness penalty
    solver.Minimize(
        lambda_cost * family_cost
        + lambda_imbalance * racial_imbalance
        + lambda_compactness * compactness_penalty
    )

    # Solve the model
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found.")
    elif status == pywraplp.Solver.FEASIBLE:
        print("Feasible solution found (may not be optimal).")
    else:
        print("No solution found.")
        return None

    # Collect the assignments
    assignments = []
    for block in block_names:
        for school in school_names:
            if x[(block, school)].solution_value() > 0.5:
                assignments.append({"Tract": block, "School": school})

    df_assignments = pd.DataFrame(assignments)

    # Merge assignments with block data
    df_results = pd.merge(df_assignments, df_blocks, on="Tract", how="left")

    # Calculate total cost components
    total_cost = solver.Objective().Value()
    family_cost_value = family_cost.solution_value()
    racial_imbalance_value = racial_imbalance.solution_value()
    compactness_penalty_value = compactness_penalty.solution_value()

    return {
        "assignments": df_results,
        "total_cost": total_cost,
        "family_cost": family_cost_value,
        "racial_imbalance": racial_imbalance_value,
        "compactness_penalty": compactness_penalty_value,
    }
