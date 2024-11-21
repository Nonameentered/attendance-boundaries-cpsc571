# streamlit_app.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from optimize import load_data, solve_optimization
from visualize import (
    plot_racial_composition,
    plot_school_entropies,
    plot_interactive_map,
)
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Howard County School District Optimizer", layout="wide")

# Title
st.title("Howard County School District Boundary Optimizer")

# Sidebar for user inputs
st.sidebar.header("Optimization Parameters")

lambda_cost = st.sidebar.slider("Lambda Cost", 0.0, 2.0, 1.0, 0.1)
lambda_imbalance = st.sidebar.slider("Lambda Racial Imbalance", 0.0, 2.0, 1.0, 0.1)
lambda_compactness = st.sidebar.slider("Lambda Compactness", 0.0, 2.0, 1.0, 0.1)

solver_time_limit = st.sidebar.slider("Solver Time Limit (seconds)", 10, 300, 30, 10)
capacity_slack = st.sidebar.slider("Capacity Slack", 1.0, 1.5, 1.05, 0.05)

# Load data
data_path = "data"  # Adjust if necessary
with st.spinner("Loading data..."):
    data = load_data(data_path)

# Optimize button
if st.button("Run Optimization"):
    with st.spinner("Running optimization... This may take a while..."):
        result = solve_optimization(
            data,
            lambda_cost=lambda_cost,
            lambda_imbalance=lambda_imbalance,
            lambda_compactness=lambda_compactness,
            solver_time_limit=solver_time_limit * 1000,  # Convert to milliseconds
            capacity_slack=capacity_slack,
        )

    if result is None:
        st.error("No solution found with the given parameters.")
    else:
        df_results = result["assignments"]
        total_cost = result["total_cost"]
        family_cost_value = result["family_cost"]
        racial_imbalance_value = result["racial_imbalance"]
        compactness_penalty_value = result["compactness_penalty"]

        st.success("Optimization Completed!")

        # Display Metrics
        st.subheader("Optimization Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Objective Value", f"{total_cost:,.2f}")
        col2.metric("Total Family Cost", f"{family_cost_value:,.2f}")
        col3.metric("Total Racial Imbalance", f"{racial_imbalance_value:,.2f}")
        col4.metric("Total Compactness Penalty", f"{compactness_penalty_value:,.2f}")

        # Calculate Racial Composition
        school_compositions = {}
        school_names = list(data["school_names"])
        for school in school_names:
            school_data = df_results[df_results["School"] == school]
            total_population = school_data["Population"].sum()
            white_population = school_data["White"].sum()
            black_population = school_data["Black"].sum()
            asian_population = school_data["Asian"].sum()
            total_race_population = (
                white_population + black_population + asian_population
            )
            school_compositions[school] = {
                "White%": (
                    white_population / total_race_population
                    if total_race_population > 0
                    else 0
                ),
                "Black%": (
                    black_population / total_race_population
                    if total_race_population > 0
                    else 0
                ),
                "Asian%": (
                    asian_population / total_race_population
                    if total_race_population > 0
                    else 0
                ),
            }

        # Racial Composition Visualization
        st.subheader("Racial Composition at Each School")
        fig1 = plot_racial_composition(df_results, school_compositions)
        st.pyplot(fig1)

        # Calculate and Display Entropy
        def compute_theil_index(distribution):
            distribution = {k: v if v > 0 else 1e-10 for k, v in distribution.items()}
            total = sum(distribution.values())
            theil_index = sum(
                [p / total * np.log(p / total) for p in distribution.values()]
            )
            return -theil_index

        school_entropies = {}
        for school, comp in school_compositions.items():
            entropy = compute_theil_index(comp)
            school_entropies[school] = entropy

        average_entropy = np.mean(list(school_entropies.values()))

        st.subheader("Entropy at Each School")
        st.write(f"**Average School Entropy (Theil Index):** {average_entropy:.4f}")
        fig2 = plot_school_entropies(school_entropies)
        st.pyplot(fig2)

        # Interactive Map
        st.subheader("Map of Tracts Assigned to Schools")
        gdf_schools = data["gdf_schools"]
        fig3 = plot_interactive_map(df_results, gdf_schools)
        st.plotly_chart(fig3, use_container_width=True)

        # Additional Metrics: Envy Count
        envy_count = 0
        for idx, block_row in df_results.iterrows():
            block = block_row["Tract"]
            assigned_school = block_row["School"]
            assigned_distance = data["dist"][block][assigned_school]
            for school in school_names:
                if school != assigned_school:
                    other_distance = data["dist"][block][school]
                    if other_distance + 1e-6 < assigned_distance:
                        envy_count += 1
        st.subheader("Envy Count")
        st.write(f"**Total Envy Count:** {envy_count}")

        # Optional: Download Results
        st.subheader("Download Assignment Results")
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="district_assignments.csv",
            mime="text/csv",
        )
