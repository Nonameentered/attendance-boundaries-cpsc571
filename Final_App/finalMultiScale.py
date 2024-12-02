import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely import wkt
import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from tqdm import tqdm
import warnings
import seaborn as sns
import time
import matplotlib
matplotlib.use('Agg')

import os

# Define the path for saving the files (Downloads folder)
download_folder = 'static/downloads'  # Adjust this path as needed
os.makedirs(download_folder, exist_ok=True)  # Ensure the folder exists

def analyze(schools_file, tracts_file, race_file):
    
    warnings.filterwarnings('ignore')

    solver_time_limit = 60000  # Time limit for the solver in milliseconds (60 sec)
    capacity_slack = 1.05      # Slack for capacity constraints - 5% overcapacity allowed

    df_schools = pd.read_csv(schools_file)

    df_schools['geometry'] = df_schools['geom'].apply(wkt.loads)
    gdf_schools = gpd.GeoDataFrame(df_schools, geometry='geometry')

    gdf_schools.head()

    df_tracts = pd.read_csv(tracts_file)

    df_tracts['geometry'] = df_tracts['geom'].apply(wkt.loads)
    gdf_tracts = gpd.GeoDataFrame(df_tracts, geometry='geometry')

    df_tracts['Tract'] = df_tracts['Tract'].astype(str)

    gdf_tracts.head()

    df_race = pd.read_csv(race_file)

    df_race.columns = df_race.iloc[0]  # Set the first row as header
    df_race = df_race[1:]              # Remove the header row from data
    df_race['Tract'] = df_race['Geography'].apply(lambda x: x[-11:])
    df_race['Tract'] = df_race['Tract'].astype(str)

    df_blocks = pd.merge(df_tracts, df_race, on='Tract', how='left')

    df_blocks.head()

    df_blocks['adj_population'] = pd.to_numeric(df_blocks['adj_population'], errors='coerce').fillna(0)
    df_blocks['Population'] = df_blocks['adj_population']

    df_blocks['Population'] = df_blocks['Population'].apply(lambda x: max(x, 1e-6))

    race_columns = [
        ' !!Total:!!Owner occupied:!!Householder who is White alone',
        ' !!Total:!!Owner occupied:!!Householder who is Black or African American alone',
        ' !!Total:!!Owner occupied:!!Householder who is Asian alone',
        ' !!Total:!!Renter occupied:!!Householder who is White alone',
        ' !!Total:!!Renter occupied:!!Householder who is Black or African American alone',
        ' !!Total:!!Renter occupied:!!Householder who is Asian alone'
    ]

    for col in race_columns:
        df_blocks[col] = pd.to_numeric(df_blocks[col], errors='coerce').fillna(0)

    df_blocks['White'] = (
        df_blocks[' !!Total:!!Owner occupied:!!Householder who is White alone'] +
        df_blocks[' !!Total:!!Renter occupied:!!Householder who is White alone']
    )
    df_blocks['Black'] = (
        df_blocks[' !!Total:!!Owner occupied:!!Householder who is Black or African American alone'] +
        df_blocks[' !!Total:!!Renter occupied:!!Householder who is Black or African American alone']
    )
    df_blocks['Asian'] = (
        df_blocks[' !!Total:!!Owner occupied:!!Householder who is Asian alone'] +
        df_blocks[' !!Total:!!Renter occupied:!!Householder who is Asian alone']
    )

    for race in ['White', 'Black', 'Asian']:
        df_blocks[race] = df_blocks[race].apply(lambda x: max(x, 1e-6))

    df_blocks.head()

    total_white = df_blocks['White'].sum()
    total_black = df_blocks['Black'].sum()
    total_asian = df_blocks['Asian'].sum()
    total_race_population = total_white + total_black + total_asian

    if total_race_population == 0:
        total_race_population = 1e-6

    P_k = {
        'White': total_white / total_race_population,
        'Black': total_black / total_race_population,
        'Asian': total_asian / total_race_population
    }

    print("Overall Racial Proportions:")
    for race, proportion in P_k.items():
        print(f"{race}: {proportion:.4f}")

    gdf_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry')
    gdf_blocks['centroid'] = gdf_blocks['geometry'].centroid

    dist = {}
    max_distance = 0  # To calculate Big M for compactness constraints
    for idx_block, block in tqdm(gdf_blocks.iterrows(), total=gdf_blocks.shape[0], desc='Calculating distances'):
        block_name = block['Tract']
        dist[block_name] = {}
        block_centroid = block['centroid']
        for idx_school, school in gdf_schools.iterrows():
            school_name = school['Name']
            school_point = school['geometry']
            distance = block_centroid.distance(school_point)
            dist[block_name][school_name] = distance
            if distance > max_distance:
                max_distance = distance

    BigM = max_distance + 1

    epsilon = 1e-6
    max_distance = max([max(dist_row.values()) for dist_row in dist.values()])

    school_names = list(gdf_schools['Name'])
    block_names = list(gdf_blocks['Tract'])

    valuations = {}
    for school in school_names:
        valuations[school] = {}
        for block in block_names:
            distance = dist[block][school]
            valuations[school][block] = max_distance - distance + epsilon  # Avoid zero valuations


    def compute_metrics(df_results):
        # Calculate total family cost
        family_cost = 0
        for idx, row in df_results.iterrows():
            block = row['Tract']
            school = row['School']
            distance = dist[block][school]
            population = row['Population']
            family_cost += population * distance

        # Calculate racial imbalance
        racial_imbalance = 0
        for school in school_names:
            school_data = df_results[df_results['School'] == school]
            n_j = school_data['Population'].sum()
            racial_imbalance_school = 0
            for race in ['White', 'Black', 'Asian']:
                n_jk = school_data[race].sum()
                expected = P_k[race] * n_j
                racial_imbalance_school += abs(n_jk - expected)
            racial_imbalance += racial_imbalance_school

        # Calculate compactness penalty (max distance to assigned blocks)
        compactness_penalty = 0
        for school in school_names:
            school_data = df_results[df_results['School'] == school]
            if school_data.empty:
                continue
            max_distance_school = school_data.apply(lambda row: dist[row['Tract']][school], axis=1).max()
            compactness_penalty += max_distance_school

        # Calculate entropy
        school_entropies = {}
        for school in school_names:
            school_data = df_results[df_results['School'] == school]
            total_population = school_data['Population'].sum()
            white_population = school_data['White'].sum()
            black_population = school_data['Black'].sum()
            asian_population = school_data['Asian'].sum()
            total_race_population = white_population + black_population + asian_population
            if total_race_population == 0:
                continue
            proportions = {
                'White': white_population / total_race_population,
                'Black': black_population / total_race_population,
                'Asian': asian_population / total_race_population
            }
            entropy = -sum([p * np.log(p) if p > 0 else 0 for p in proportions.values()])
            school_entropies[school] = entropy

        average_entropy = np.mean(list(school_entropies.values()))

        # Compute envy count
        envy_count = 0
        for idx, row in df_results.iterrows():
            block = row['Tract']
            assigned_school = row['School']
            assigned_distance = dist[block][assigned_school]
            for school in school_names:
                if school != assigned_school:
                    other_distance = dist[block][school]
                    if other_distance + 1e-6 < assigned_distance:
                        envy_count += 1
                        break  # Count each block only once

        return {
            'family_cost': family_cost,
            'racial_imbalance': racial_imbalance,
            'compactness_penalty': compactness_penalty,
            'average_entropy': average_entropy,
            'envy_count': envy_count
        }


    from collections import defaultdict, deque

    def eefx_allocation(valuations, capacity_per_school):
        # Initialize allocation and unallocated items
        allocation = {school: [] for school in school_names}
        unallocated_blocks = set(block_names)

        # Initialize capacities
        remaining_capacity = {school: capacity_per_school for school in school_names}

        # While there are unallocated blocks
        while unallocated_blocks:
            # Each school selects its most valued unallocated block
            proposals = {}
            proposed_blocks = set()
            for school in school_names:
                if remaining_capacity[school] > 0:
                    # Get the unallocated blocks sorted by valuation
                    preferred_blocks = sorted(
                        unallocated_blocks,
                        key=lambda block: -valuations[school][block]
                    )
                    # Find a block that fits in the remaining capacity and hasn't been proposed yet
                    for block in preferred_blocks:
                        if block in proposed_blocks:
                            continue  # Skip if block is already proposed
                        block_population = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
                        if remaining_capacity[school] - block_population >= 0:
                            proposals[school] = block
                            proposed_blocks.add(block)
                            break

            # If no proposals, break
            if not proposals:
                break

            # Build envy graph
            envy_graph = defaultdict(list)
            for school_i in proposals:
                for school_j in proposals:
                    if school_i != school_j:
                        # If school_i values school_j's proposed block more than its own
                        if valuations[school_i][proposals[school_j]] > valuations[school_i][proposals[school_i]]:
                            envy_graph[school_i].append(school_j)

            # Detect cycles in envy graph
            def find_cycle():
                visited = set()
                stack = []

                def dfs(school):
                    if school in stack:
                        return stack[stack.index(school):]
                    if school in visited:
                        return None
                    visited.add(school)
                    stack.append(school)
                    for neighbor in envy_graph[school]:
                        cycle = dfs(neighbor)
                        if cycle:
                            return cycle
                    stack.pop()
                    return None

                for school in proposals:
                    cycle = dfs(school)
                    if cycle:
                        return cycle
                return None

            cycle = find_cycle()
            if cycle:
                # Allocate blocks along the cycle
                for school in cycle:
                    block = proposals[school]
                    allocation[school].append(block)
                    unallocated_blocks.remove(block)
                    block_population = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
                    remaining_capacity[school] -= block_population
            else:
                # Allocate proposed blocks
                for school, block in proposals.items():
                    allocation[school].append(block)
                    unallocated_blocks.remove(block)
                    block_population = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
                    remaining_capacity[school] -= block_population

        return allocation

    lambda_distance = 0.5
    lambda_racial = 0.3
    lambda_capacity = 0.2

    # Maximum possible racial imbalance for normalization
    max_possible_imbalance = 1.0  # Since proportions range from 0 to 1

    # Maximum distance for normalization (should be defined earlier in your code)
    # If not defined, compute it here
    max_distance = max([max(dist_row.values()) for dist_row in dist.values()])

    def compute_composite_valuations(allocation, remaining_capacity, unallocated_blocks):
        valuations = {}
        for school in school_names:
            valuations[school] = {}
            # Get current school allocation
            school_blocks = allocation[school]
            school_data = df_blocks[df_blocks['Tract'].isin(school_blocks)]
            n_s = school_data['Population'].sum()
            n_s_k = {race: school_data[race].sum() for race in ['White', 'Black', 'Asian']}
            for block in unallocated_blocks:
                # Distance valuation
                distance = dist[block][school]
                V_distance = (max_distance - distance) / max_distance

                # Racial balance valuation
                n_b = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
                n_b_k = {race: df_blocks.loc[df_blocks['Tract'] == block, race].values[0] for race in ['White', 'Black', 'Asian']}
                n_total = n_s + n_b
                n_s_k_new = {race: n_s_k[race] + n_b_k[race] for race in ['White', 'Black', 'Asian']}
                racial_imbalance_after = sum([abs((n_s_k_new[race]/n_total if n_total > 0 else 0) - P_k[race]) for race in ['White', 'Black', 'Asian']])
                V_racial = - (racial_imbalance_after) / (3 * max_possible_imbalance)  # Divided by 3 since there are 3 races

                # Capacity impact valuation
                remaining_cap = remaining_capacity[school]
                V_capacity = 1 if remaining_cap >= n_b else 0  # 1 if within capacity, 0 otherwise

                # Composite valuation
                valuations[school][block] = (lambda_distance * V_distance +
                                            lambda_racial * V_racial +
                                            lambda_capacity * V_capacity)
        return valuations

    def eefx_allocation_composite(remaining_capacity):
        # Initialize allocation and unallocated items
        allocation = {school: [] for school in school_names}
        unallocated_blocks = set(block_names)

        # While there are unallocated blocks
        while unallocated_blocks:
            # Compute composite valuations
            valuations = compute_composite_valuations(allocation, remaining_capacity, unallocated_blocks)

            # Each school selects its most valued unallocated block
            proposals = {}
            proposed_blocks = set()
            for school in school_names:
                if remaining_capacity[school] > 0:
                    # Get the unallocated blocks sorted by valuation
                    preferred_blocks = sorted(
                        unallocated_blocks,
                        key=lambda block: -valuations[school][block]
                    )
                    # Find a block that fits in the remaining capacity and hasn't been proposed yet
                    for block in preferred_blocks:
                        if block in proposed_blocks:
                            continue  # Skip if block is already proposed
                        block_population = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
                        if remaining_capacity[school] - block_population >= 0:
                            proposals[school] = block
                            proposed_blocks.add(block)
                            break

            # If no proposals, break
            if not proposals:
                break

            # Build envy graph
            envy_graph = defaultdict(list)
            for school_i in proposals:
                for school_j in proposals:
                    if school_i != school_j:
                        # If school_i values school_j's proposed block more than its own
                        if valuations[school_i][proposals[school_j]] > valuations[school_i][proposals[school_i]]:
                            envy_graph[school_i].append(school_j)

            # Detect cycles in envy graph
            def find_cycle():
                visited = set()
                stack = []

                def dfs(school):
                    if school in stack:
                        return stack[stack.index(school):]
                    if school in visited:
                        return None
                    visited.add(school)
                    stack.append(school)
                    for neighbor in envy_graph[school]:
                        cycle = dfs(neighbor)
                        if cycle:
                            return cycle
                    stack.pop()
                    return None

                for school in proposals:
                    cycle = dfs(school)
                    if cycle:
                        return cycle
                return None

            cycle = find_cycle()
            if cycle:
                # Allocate blocks along the cycle
                for school in cycle:
                    block = proposals[school]
                    allocation[school].append(block)
                    unallocated_blocks.remove(block)
                    block_population = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
                    remaining_capacity[school] -= block_population
            else:
                # Allocate proposed blocks
                for school, block in proposals.items():
                    allocation[school].append(block)
                    unallocated_blocks.remove(block)
                    block_population = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
                    remaining_capacity[school] -= block_population

        return allocation


    def solve_optimization(lambda_cost, lambda_imbalance, lambda_compactness):
        # Create optimization model
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print('Solver not available.')
            return None

        # Set solver parameters for time limit
        solver.SetTimeLimit(solver_time_limit)  # Time limit in milliseconds

        # Decision variables
        x = {}
        block_names = list(gdf_blocks['Tract'])
        school_names = list(gdf_schools['Name'])

        for block in block_names:
            for school in school_names:
                x[(block, school)] = solver.IntVar(0, 1, f'x_{block}_{school}')

        # Auxiliary variables for racial imbalance
        d = {}
        for school in school_names:
            for race in ['White', 'Black', 'Asian']:
                d[(school, race)] = solver.NumVar(0, solver.infinity(), f'd_{school}_{race}')

        # Variables for total assigned populations
        n_j = {}
        n_jk = {}
        for school in school_names:
            n_j[school] = solver.NumVar(0, solver.infinity(), f'n_{school}')
            for race in ['White', 'Black', 'Asian']:
                n_jk[(school, race)] = solver.NumVar(0, solver.infinity(), f'n_{school}_{race}')

        # Variables for compactness
        M = {}
        for school in school_names:
            M[school] = solver.NumVar(0, solver.infinity(), f'M_{school}')

        # Constraints

        # Assignment Constraint: Each block must be assigned to exactly one school
        for block in block_names:
            solver.Add(solver.Sum([x[(block, school)] for school in school_names]) == 1)

        # Capacity Constraint: Total assigned population should not exceed capacity (with slack)
        total_population = df_blocks['Population'].sum()
        num_schools = len(school_names)
        capacity_per_school = capacity_slack * (total_population / num_schools)

        for school in school_names:
            solver.Add(
                solver.Sum([
                    df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0] * x[(block, school)]
                    for block in block_names
                ]) <= capacity_per_school
            )

        # Define n_j and n_jk (Total assigned populations)
        for school in school_names:
            # Total population assigned to school
            solver.Add(n_j[school] == solver.Sum([
                df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0] * x[(block, school)]
                for block in block_names
            ]))
            for race in ['White', 'Black', 'Asian']:
                # Total population of race assigned to school
                solver.Add(n_jk[(school, race)] == solver.Sum([
                    df_blocks.loc[df_blocks['Tract'] == block, race].values[0] * x[(block, school)]
                    for block in block_names
                ]))
                # Racial Imbalance Constraints: d_{jk} >= |n_{jk} - P_k * n_j|
                solver.Add(d[(school, race)] >= n_jk[(school, race)] - P_k[race] * n_j[school])
                solver.Add(d[(school, race)] >= - (n_jk[(school, race)] - P_k[race] * n_j[school]))
                solver.Add(d[(school, race)] >= 0)

        # Compactness Constraints: M_j >= d_{ij} - (1 - x_{ij}) * BigM
        for school in school_names:
            for block in block_names:
                distance = dist[block][school]
                solver.Add(M[school] >= distance - (1 - x[(block, school)]) * BigM)

        # Compute Family Cost (Total travel distance for students)
        family_cost_terms = []
        for block in block_names:
            block_population = df_blocks.loc[df_blocks['Tract'] == block, 'Population'].values[0]
            for school in school_names:
                distance_to_school = dist[block][school]
                cost = block_population * distance_to_school * x[(block, school)]
                family_cost_terms.append(cost)

        family_cost = solver.Sum(family_cost_terms)

        # Racial Imbalance Term
        racial_imbalance = solver.Sum([d[(school, race)] for school in school_names for race in ['White', 'Black', 'Asian']])

        # Compactness Penalty Term
        compactness_penalty = solver.Sum([M[school] for school in school_names])

        # Objective Function: Minimize weighted sum of family cost, racial imbalance, and compactness penalty
        solver.Minimize(lambda_cost * family_cost + lambda_imbalance * racial_imbalance + lambda_compactness * compactness_penalty)

        # Solve the model
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print('Optimal solution found.')
        elif status == pywraplp.Solver.FEASIBLE:
            print('Feasible solution found (may not be optimal).')
        else:
            print('No solution found.')
            return None

        # Collect the assignments
        assignments = []
        for block in block_names:
            for school in school_names:
                if x[(block, school)].solution_value() > 0.5:
                    assignments.append({'Tract': block, 'School': school})

        df_assignments = pd.DataFrame(assignments)

        # Merge assignments with block data
        df_results = pd.merge(df_assignments, df_blocks, on='Tract', how='left')

        # Calculate total cost components
        total_cost = solver.Objective().Value()
        family_cost_value = family_cost.solution_value()
        racial_imbalance_value = racial_imbalance.solution_value()
        compactness_penalty_value = compactness_penalty.solution_value()

        return {
            'assignments': df_results,
            'total_cost': total_cost,
            'family_cost': family_cost_value,
            'racial_imbalance': racial_imbalance_value,
            'compactness_penalty': compactness_penalty_value
        }

    # Run EEFX Algorithm
    start_time = time.time()
    total_population = df_blocks['Population'].sum()
    num_schools = len(school_names)
    capacity_per_school = capacity_slack * (total_population / num_schools)
    eefx_alloc = eefx_allocation(valuations, capacity_per_school)
    eefx_time = time.time() - start_time

    eefx_assignments = []
    for school, blocks in eefx_alloc.items():
        for block in blocks:
            eefx_assignments.append({'Tract': block, 'School': school})

    df_eefx_assignments = pd.DataFrame(eefx_assignments)

    df_eefx_results = pd.merge(df_eefx_assignments, df_blocks, on='Tract', how='left')

    # Initialize remaining capacity
    start_time = time.time()
    total_population = df_blocks['Population'].sum()
    num_schools = len(school_names)
    capacity_per_school = capacity_slack * (total_population / num_schools)
    remaining_capacity = {school: capacity_per_school for school in school_names}

    # Run EEFX allocation with composite valuations
    eefx_alloc_composite = eefx_allocation_composite(remaining_capacity)
    eefx_composite_time = time.time() - start_time

    # Prepare results DataFrame
    eefx_assignments_composite = []
    for school, blocks in eefx_alloc_composite.items():
        for block in blocks:
            eefx_assignments_composite.append({'Tract': block, 'School': school})

    df_eefx_assignments_composite = pd.DataFrame(eefx_assignments_composite)

    # Merge with block data
    df_eefx_results_composite = pd.merge(df_eefx_assignments_composite, df_blocks, on='Tract', how='left')

    lambda_cost = 1
    lambda_imbalance = 1
    lambda_compactness = 1

    start_time = time.time()
    result = solve_optimization(lambda_cost, lambda_imbalance, lambda_compactness)
    milp_time = time.time() - start_time

    if result is None:
        print("No solution found with the initial parameters.")
    else:
        df_results = result['assignments']
        total_cost = result['total_cost']
        family_cost_value = result['family_cost']
        racial_imbalance_value = result['racial_imbalance']
        compactness_penalty_value = result['compactness_penalty']

        print(f"Total Objective Value: {total_cost}")
        print(f"Total Family Cost: {family_cost_value}")
        print(f"Total Racial Imbalance: {racial_imbalance_value}")
        print(f"Total Compactness Penalty: {compactness_penalty_value}")

    eefx_metrics = compute_metrics(df_eefx_results)

    # Compute metrics for Adjusted EEFX allocation
    eefx_composite_metrics = compute_metrics(df_eefx_results_composite)

    # Compute metrics for MILP allocation
    milp_metrics = compute_metrics(df_results)


    gdf_eefx_results = gpd.GeoDataFrame(df_eefx_results, geometry='geometry')
    gdf_eefx_results_composite = gpd.GeoDataFrame(df_eefx_results_composite, geometry='geometry')
    gdf_milp_results = gpd.GeoDataFrame(df_results, geometry='geometry')

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    gdf_eefx_results.plot(column='School', ax=axs[0], legend=True)
    gdf_schools.plot(ax=axs[0], color='red', markersize=50)
    axs[0].set_title('EEF Allocation')

    gdf_eefx_results_composite.plot(column='School', ax=axs[1], legend=True)
    gdf_schools.plot(ax=axs[1], color='red', markersize=50)
    axs[1].set_title('Adjusted EEF Allocation')

    gdf_milp_results.plot(column='School', ax=axs[2], legend=True)
    gdf_schools.plot(ax=axs[2], color='red', markersize=50)
    axs[2].set_title('MILP Allocation')

    plot_path = os.path.join(download_folder, 'allocation_plots.png')
    fig.savefig(plot_path)

    # Close the plot to avoid any further changes
    plt.close(fig)

    metrics = ['Family Cost', 'Racial Imbalance', 'Compactness Penalty', 'Average Entropy', 'Envy Count', 'Computation Time (s)']
    milp_values = [milp_metrics['family_cost'], milp_metrics['racial_imbalance'],
                milp_metrics['compactness_penalty'], milp_metrics['average_entropy'], milp_metrics['envy_count'], milp_time]
    eefx_values = [eefx_metrics['family_cost'], eefx_metrics['racial_imbalance'],
                eefx_metrics['compactness_penalty'], eefx_metrics['average_entropy'], eefx_metrics['envy_count'], eefx_time]
    eefx_composite_values = [eefx_composite_metrics['family_cost'], eefx_composite_metrics['racial_imbalance'],
                            eefx_composite_metrics['compactness_penalty'], eefx_composite_metrics['average_entropy'], eefx_composite_metrics['envy_count'], eefx_composite_time]


    data = {
        'Metric': metrics,
        'MILP Allocation': milp_values,
        'EEF Allocation': eefx_values,
        'Adjusted EEF Allocation': eefx_composite_values
    }

    df_metrics = pd.DataFrame(data)

    # Set the 'Metric' column as the index
    df_metrics.set_index('Metric', inplace=True)

    # Transpose the DataFrame to have methods as rows and metrics as columns
    df_metrics = df_metrics.transpose()

    # Plotting metrics in subplots
    import matplotlib.ticker as mtick

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        methods = ['MILP Allocation', 'EEF Allocation', 'Adjusted EEF Allocation']
        values = [milp_values[idx], eefx_values[idx], eefx_composite_values[idx]]
        axes[idx].bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[idx].set_title(metric)
        axes[idx].set_xticklabels(methods, rotation=45)
        # Format y-axis
        if metric in ['Average Entropy', 'Computation Time (s)']:
            axes[idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        else:
            axes[idx].yaxis.get_major_locator().set_params(integer=True)
        axes[idx].set_ylabel(metric)

    # Remove any empty subplots
    if len(metrics) < len(axes):
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plot_path = os.path.join(download_folder, 'metrics_plot.png')
    fig.savefig(plot_path)

    # Close the plot to avoid any further changes
    plt.close(fig)

    lambda_values = [0.0, 0.5, 1.0]  # Values for lambdas
    fixed_capacity_slack = 1.05      # Fixed capacity slack

    milp_grid_search_results = []

    for lambda_cost in tqdm(lambda_values, desc='Lambda Cost Values'):
        for lambda_imbalance in lambda_values:
            for lambda_compactness in lambda_values:
                # Skip case where all lambdas are zero
                if lambda_cost == 0 and lambda_imbalance == 0 and lambda_compactness == 0:
                    continue
                print(f"\nRunning MILP optimization with lambda_cost = {lambda_cost:.2f}, lambda_imbalance = {lambda_imbalance:.2f}, lambda_compactness = {lambda_compactness:.2f}")
                start_time = time.time()
                result = solve_optimization(lambda_cost, lambda_imbalance, lambda_compactness)
                computation_time = time.time() - start_time
                if result is not None:
                    df_results = result['assignments']
                    metrics = compute_metrics(df_results)
                    milp_grid_search_results.append({
                        'method': 'MILP',
                        'lambda_cost': lambda_cost,
                        'lambda_imbalance': lambda_imbalance,
                        'lambda_compactness': lambda_compactness,
                        'total_cost': result['total_cost'],
                        'family_cost': metrics['family_cost'],
                        'racial_imbalance': metrics['racial_imbalance'],
                        'compactness_penalty': metrics['compactness_penalty'],
                        'average_entropy': metrics['average_entropy'],
                        'envy_count': metrics['envy_count'],
                        'computation_time': computation_time
                    })
                else:
                    print("No solution found.")


    # Adjusted EEFX doesn't have lambdas but we can vary the weights in the composite valuations
    lambda_distance_values = [0.2, 0.5, 0.8]
    lambda_racial_values = [0.2, 0.5, 0.8]
    lambda_capacity_values = [0.0, 0.2]

    eefx_grid_search_results = []

    for lambda_distance in lambda_distance_values:
        for lambda_racial in lambda_racial_values:
            for lambda_capacity in lambda_capacity_values:
                # Ensure weights sum to 1
                total_lambda = lambda_distance + lambda_racial + lambda_capacity
                if total_lambda == 0:
                    continue
                lambda_distance_normalized = lambda_distance / total_lambda
                lambda_racial_normalized = lambda_racial / total_lambda
                lambda_capacity_normalized = lambda_capacity / total_lambda
                print(f"\nRunning Adjusted EEFX allocation with weights: distance={lambda_distance_normalized:.2f}, racial={lambda_racial_normalized:.2f}, capacity={lambda_capacity_normalized:.2f}")
                # Update weights
                lambda_distance = lambda_distance_normalized
                lambda_racial = lambda_racial_normalized
                lambda_capacity = lambda_capacity_normalized

                # Initialize remaining capacity
                start_time = time.time()
                total_population = df_blocks['Population'].sum()
                num_schools = len(school_names)
                capacity_per_school = capacity_slack * (total_population / num_schools)
                remaining_capacity = {school: capacity_per_school for school in school_names}

                # Run EEFX allocation with composite valuations
                eefx_alloc_composite = eefx_allocation_composite(remaining_capacity)
                computation_time = time.time() - start_time

                # Prepare results DataFrame
                eefx_assignments_composite = []
                for school, blocks in eefx_alloc_composite.items():
                    for block in blocks:
                        eefx_assignments_composite.append({'Tract': block, 'School': school})

                df_eefx_assignments_composite = pd.DataFrame(eefx_assignments_composite)

                # Merge with block data
                df_eefx_results_composite = pd.merge(df_eefx_assignments_composite, df_blocks, on='Tract', how='left')

                metrics = compute_metrics(df_eefx_results_composite)

                eefx_grid_search_results.append({
                    'method': 'Adjusted EEF',
                    'lambda_distance': lambda_distance,
                    'lambda_racial': lambda_racial,
                    'lambda_capacity': lambda_capacity,
                    'family_cost': metrics['family_cost'],
                    'racial_imbalance': metrics['racial_imbalance'],
                    'compactness_penalty': metrics['compactness_penalty'],
                    'average_entropy': metrics['average_entropy'],
                    'envy_count': metrics['envy_count'],
                    'computation_time': computation_time
                })

    df_milp_results = pd.DataFrame(milp_grid_search_results)
    df_eefx_results = pd.DataFrame(eefx_grid_search_results)


    plt.figure(figsize=(10, 6))
    for lambda_compactness in lambda_values:
        df_subset = df_milp_results[df_milp_results['lambda_compactness'] == lambda_compactness]
        plt.plot(df_subset['lambda_cost'], df_subset['family_cost'], marker='o', label=f'Compactness Lambda = {lambda_compactness}')
    plt.xlabel('Lambda Cost')
    plt.ylabel('Family Cost')
    plt.title('MILP: Family Cost vs Lambda Cost')
    plt.legend()
    plot_path_1 = os.path.join(download_folder, 'family_cost_vs_lambda_cost.png')
    plt.savefig(plot_path_1)
    plt.close()  # Close the figure to avoid overlapping plots

    plt.figure(figsize=(10, 6))
    for lambda_cost in lambda_values:
        df_subset = df_milp_results[df_milp_results['lambda_cost'] == lambda_cost]
        plt.plot(df_subset['lambda_imbalance'], df_subset['racial_imbalance'], marker='o', label=f'Cost Lambda = {lambda_cost}')
    plt.xlabel('Lambda Imbalance')
    plt.ylabel('Racial Imbalance')
    plt.title('MILP: Racial Imbalance vs Lambda Imbalance')
    plt.legend()
    plot_path_2 = os.path.join(download_folder, 'racial_imbalance_vs_lambda_imbalance.png')
    plt.savefig(plot_path_2)
    plt.close()

    plt.figure(figsize=(10, 6))
    for lambda_capacity in lambda_capacity_values:
        df_subset = df_eefx_results[df_eefx_results['lambda_capacity'] == lambda_capacity]
        plt.plot(df_subset['lambda_distance'], df_subset['family_cost'], marker='o', label=f'Capacity Weight = {lambda_capacity}')
    plt.xlabel('Lambda Distance')
    plt.ylabel('Family Cost')
    plt.title('Adjusted EEFX: Family Cost vs Lambda Distance')
    plt.legend()

    # Save the first plot
    plot_path_1 = os.path.join(download_folder, 'adjusted_eefx_family_cost_vs_lambda_distance.png')
    plt.savefig(plot_path_1)
    plt.close()  # Close the plot after saving

    # Second scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_milp_results, x='average_entropy', y='envy_count', hue='lambda_cost', style='lambda_imbalance', size='lambda_compactness', sizes=(50, 200), palette='viridis')
    plt.xlabel('Average Entropy (Theil Index)')
    plt.ylabel('Envy Count')
    plt.title('MILP Allocations: Entropy vs Envy Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the second plot
    plot_path_2 = os.path.join(download_folder, 'milp_entropy_vs_envy_count.png')
    plt.savefig(plot_path_2)
    plt.close()  # Close the plot after saving

    # Third scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_eefx_results, x='average_entropy', y='envy_count', hue='lambda_distance', style='lambda_racial', size='lambda_capacity', sizes=(50, 200), palette='coolwarm')
    plt.xlabel('Average Entropy (Theil Index)')
    plt.ylabel('Envy Count')
    plt.title('Adjusted EEFX Allocations: Entropy vs Envy Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the third plot
    plot_path_3 = os.path.join(download_folder, 'adjusted_eefx_entropy_vs_envy_count.png')
    plt.savefig(plot_path_3)
    plt.close()  # Close the plot after saving