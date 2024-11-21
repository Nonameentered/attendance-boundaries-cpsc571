from email.policy import default
from utils.header import *
import geopandas as gpd


def analyze_historical_rezonings(
    input_dir="data/derived_data/{}/",
    state_dir="{}/blocks_to_elementary.csv",
    state_codes_file="data/state_codes.csv",
    zones_years=["1314", "1516"],
    output_file="data/derived_data/{}_{}_rezonings.csv",
):

    y1 = zones_years[0]
    y2 = zones_years[1]
    sch_key = {y1: "ncessch", y2: "ncessch2"}
    dist_key = {y1: "leaid", y2: "leaid2"}
    df_states = pd.read_csv(state_codes_file)
    stats = {}
    for y in zones_years:
        stats["all_districts_{}".format(y)] = []
        stats["districts_with_>1_elem_{}".format(y)] = []
        stats["defacto_districts_{}".format(y)] = []
    stats["districts_with_school_rezonings"] = []
    stats["districts_with_district_rezonings"] = []
    stats["districts_with_new_schools"] = []

    for i in range(0, len(df_states)):
        code = df_states["abbrev"][i]
        print(code)

        try:
            dfs = {}
            for y in zones_years:
                dfs[y] = (
                    pd.read_csv(
                        input_dir.format(y) + state_dir.format(code), dtype="str"
                    )
                    .rename(columns={"ncessch": sch_key[y], "leaid": dist_key[y]})
                    .reset_index()
                )
            dfs[y2] = dfs[y2][["GEOID10", sch_key[y2], dist_key[y2], "defacto"]]

            # Determine which schools in y2 are new (i.e. weren't in the y1 list)
            df_merged = pd.merge(dfs[y1], dfs[y2], on="GEOID10", how="inner").replace(
                np.nan, "", regex=True
            )
            y1_schools = set(df_merged[sch_key[y1]].tolist())
            y2_schools = set(df_merged[sch_key[y2]].tolist())

            new_schools = list(y2_schools - y1_schools)
            is_new_school = [
                dfs[y2][sch_key[y2]][i] in new_schools for i in range(0, len(dfs[y2]))
            ]
            dfs[y1]["is_new_school"] = [False for i in range(0, len(dfs[y1]))]
            dfs[y2]["is_new_school"] = is_new_school

            # Determine blocks that got rezoned from one school or district to another
            block_rezoned_school = []
            block_rezoned_district = []
            df_merged = pd.merge(dfs[y1], dfs[y2], on="GEOID10", how="left").replace(
                np.nan, "", regex=True
            )

            for i in range(0, len(df_merged)):

                # Check if block was zoned to a new school in y2 compared to y1
                if (
                    df_merged[sch_key[y2]][i]
                    and df_merged[sch_key[y1]][i] != df_merged[sch_key[y2]][i]
                ):
                    block_rezoned_school.append(True)
                else:
                    block_rezoned_school.append(False)

                # Check if block was zoned to a new district in y2 compared to y1
                if (
                    df_merged[dist_key[y2]][i]
                    and df_merged[dist_key[y1]][i] != df_merged[dist_key[y2]][i]
                ):
                    block_rezoned_district.append(True)
                else:
                    block_rezoned_district.append(False)

            dfs[y1]["block_rezoned_school"] = block_rezoned_school
            dfs[y2]["block_rezoned_school"] = [False for i in range(0, len(dfs[y2]))]

            dfs[y1]["block_rezoned_district"] = block_rezoned_district
            dfs[y2]["block_rezoned_district"] = [False for i in range(0, len(dfs[y2]))]

            # Compute stats per year
            grouped_dfs = {}
            for y in zones_years:

                # Group to the school and then district level
                grouped_dfs[y] = (
                    dfs[y]
                    .groupby(sch_key[y])
                    .agg(
                        {
                            dist_key[y]: "first",
                            "is_new_school": "first",
                            sch_key[y]: "first",
                            "GEOID10": "count",
                            "block_rezoned_school": "sum",
                            "block_rezoned_district": "sum",
                            "defacto": "first",
                        }
                    )
                    .groupby(dist_key[y])
                    .agg(
                        {
                            sch_key[y]: "count",
                            "is_new_school": "sum",
                            "block_rezoned_school": "sum",
                            "block_rezoned_district": "sum",
                            "defacto": "first",
                        }
                    )
                    .rename(
                        columns={
                            "is_new_school": "num_new_schools",
                            "block_rezoned_school": "num_blocks_rezoned_schools",
                            "block_rezoned_district": "num_blocks_rezoned_districts",
                        }
                    )
                    .reset_index()
                )

                for i in range(0, len(grouped_dfs[y])):

                    # Per year, store the number of districts,
                    # number of districts with >1 elem, and
                    # number of "defacto" districts i.e. (only one school)
                    stats["all_districts_{}".format(y)].append(
                        grouped_dfs[y][dist_key[y]][i]
                    )

                    if grouped_dfs[y][sch_key[y]][i] > 1:
                        stats["districts_with_>1_elem_{}".format(y)].append(
                            grouped_dfs[y][dist_key[y]][i]
                        )
                    if grouped_dfs[y]["defacto"][i] in ["1", "Yes"]:
                        stats["defacto_districts_{}".format(y)].append(
                            grouped_dfs[y][dist_key[y]][i]
                        )

            # Store the districts that have new schools in them
            for i in range(0, len(grouped_dfs[y2])):
                num_new_schools = grouped_dfs[y2]["num_new_schools"][i]
                if num_new_schools > 0:
                    stats["districts_with_new_schools"].append(
                        (grouped_dfs[y2][dist_key[y2]][i], num_new_schools)
                    )

            for i in range(0, len(grouped_dfs[y1])):

                # Store the districts that have blocks that were re-assigned to a different school in 15/16 compared to 13/14
                num_blocks_rezoned_schools = grouped_dfs[y1][
                    "num_blocks_rezoned_schools"
                ][i]
                if num_blocks_rezoned_schools > 0:
                    stats["districts_with_school_rezonings"].append(
                        (grouped_dfs[y1][dist_key[y1]][i], num_blocks_rezoned_schools)
                    )

                # Store the districts that have blocks that were re-assigned to a different district in 15/16 compared to 13/14
                num_blocks_rezoned_districts = grouped_dfs[y1][
                    "num_blocks_rezoned_districts"
                ][i]
                if num_blocks_rezoned_districts > 0:
                    stats["districts_with_district_rezonings"].append(
                        (grouped_dfs[y1][dist_key[y1]][i], num_blocks_rezoned_districts)
                    )

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, e, exc_tb.tb_lineno)
            continue

        # break

    for k in stats:
        print(k, len(set(stats[k])))


def compute_dissimilarity_index(df_dist, key):
    vals = []
    df_dist[key] = df_dist[key].astype(float)
    df_dist["total_enrollment"] = df_dist["total_enrollment"].astype(float)
    cat_total = df_dist[key].sum()
    non_cat_total = df_dist["total_enrollment"].sum() - df_dist[key].sum()
    for i in range(0, len(df_dist)):
        vals.append(
            np.abs(
                (df_dist[key][i] / cat_total)
                - ((df_dist["total_enrollment"][i] - df_dist[key][i]) / non_cat_total)
            )
        )

    return 0.5 * np.nansum(vals)


def compute_normalized_exposure_index(df_dist, key):
    vals = []
    for i in range(0, len(df_dist)):
        vals.append(
            (df_dist[key][i] / df_dist[key].sum())
            * (df_dist[key][i] / df_dist["total_enrollment"][i])
        )

    P = df_dist[key].sum() / df_dist["total_enrollment"].sum()
    return (np.nansum(vals) - P) / (1 - P)


def compute_gini_index(df_dist, key):
    vals = []
    for i in range(0, len(df_dist)):
        for j in range(0, len(df_dist)):
            vals.append(
                (df_dist["total_enrollment"][i] * df_dist["total_enrollment"][j])
                * np.abs(
                    (df_dist[key][i] / df_dist["total_enrollment"][i])
                    - (df_dist[key][j] / df_dist["total_enrollment"][j])
                )
            )

    P = df_dist[key].sum() / df_dist["total_enrollment"].sum()
    denom = 2 * (df_dist["total_enrollment"].sum() ** 2) * P * (1 - P)
    return np.nansum(vals) / denom


def count_fraction_closed_enrollment_districts(
    sab_elem_file="data/school_boundary_shapefiles/ATTOM_boundaries_21_22_USA/school-attendance-areas-10.shp",
    school_files_dir="data/derived_data/2122",
    output_file="data/school_covariates/open_v_closed_enrollment_districts.csv",
):

    print("Loading elementary school zones")
    elem_zones = (
        gpd.read_file(sab_elem_file)
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
    is_open_enrollment = []
    for i in range(0, len(elem_zones)):
        if elem_zones["openEnroll"][i] == "Y":
            is_open_enrollment.append(1)
        else:
            is_open_enrollment.append(0)

    elem_zones["is_open_enrollment"] = is_open_enrollment

    df_districts = (
        elem_zones.groupby(["leaid"], as_index=False)
        .agg({"leaid": "first", "ncessch": "count", "is_open_enrollment": "sum"})
        .rename(
            columns={
                "ncessch": "num_schools",
                "is_open_enrollment": "num_open_enrollment_schools",
            }
        )
    )

    # Next, compute segregation values for each of those districts
    data = []
    for state in os.listdir(school_files_dir):
        data.append(
            pd.read_csv(
                os.path.join(
                    school_files_dir, state, "schools_file_for_assignment.csv"
                ),
                dtype="str",
            )
        )

    df_schools = pd.concat(data)

    df_schools["district_totenrl"] = df_schools["district_totenrl"].astype(float)
    df_schools["total_white"] = df_schools["total_white"].astype(float)
    df_schools["total_native"] = df_schools["total_native"].astype(float)
    df_schools["total_asian"] = df_schools["total_asian"].astype(float)
    df_schools["total_hispanic"] = df_schools["total_hispanic"].astype(float)
    df_schools["total_black"] = df_schools["total_black"].astype(float)

    df_schools_g = df_schools.groupby(["leaid"], as_index=False).agg(
        {
            "leaid": "first",
            "district_totenrl": "first",
            "total_white": "sum",
            "total_native": "sum",
            "total_asian": "sum",
            "total_hispanic": "sum",
            "total_black": "sum",
        }
    )

    df_districts = pd.merge(
        df_districts, df_schools_g, how="inner", on="leaid"
    ).sort_values(by="district_totenrl", ascending=False)

    # keys = {
    #     "hisp": "hispanic",
    #     "black": "black",
    #     "white": "white",
    #     "frl": "frl",
    #     "ell": "ell",
    #     "asian": "asian",
    # }

    # dissim_indices = {
    #     "leaid": [],
    #     "hisp_dissim": [],
    #     "black_dissim": [],
    #     "white_dissim": [],
    #     "frl_dissim": [],
    #     "ell_dissim": [],
    #     "asian_dissim": [],
    # }

    # district_ids = df_districts["leaid"].tolist()
    # for dist in district_ids:
    #     df_curr = df_schools[df_schools["leaid"] == dist].reset_index(drop=True)
    #     dissim_indices["leaid"].append(dist)
    #     for k in keys:
    #         try:
    #             curr = compute_dissimilarity_index(df_curr, "total_{}".format(keys[k]))
    #         except Exception as e:
    #             print(e)
    #             curr = float("nan")
    #         dissim_indices["{}_dissim".format(k)].append(curr)

    # df_dissim = pd.DataFrame(data=dissim_indices)
    # df_districts = pd.merge(df_districts, df_dissim, how="inner", on="leaid")

    # Print stuff about districts
    print(
        len(df_districts),
        len(df_districts[df_districts["num_open_enrollment_schools"] == 0]),
        len(df_districts[df_districts["num_open_enrollment_schools"] > 0]),
    )
    df_districts = df_districts[
        (df_districts["num_schools"] > 1) & (df_districts["num_schools"] <= 200)
    ].reset_index(drop=True)

    print(
        len(df_districts),
        len(df_districts[df_districts["num_open_enrollment_schools"] == 0]),
        len(df_districts[df_districts["num_open_enrollment_schools"] > 0]),
    )
    exit()
    df_districts["has_open_enrollment"] = (
        df_districts["num_open_enrollment_schools"] > 0
    )

    print(df_districts.keys())
    df_districts.to_csv(output_file)


def identify_largest_districts_with_all_closed_enrollment(
    sab_elem_file="data/school_boundary_shapefiles/ATTOM_boundaries_21_22_USA/school-attendance-areas-03.shp",
    school_files_dir="data/derived_data/2122",
    N_START=0,
    N_END=100,
    output_file="data/school_covariates/top_{}_{}_largest_districts_updated.csv",
):

    print("Loading elementary school zones")
    elem_zones = (
        gpd.read_file(sab_elem_file)
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
    is_open_enrollment = []
    for i in range(0, len(elem_zones)):
        if elem_zones["openEnroll"][i] == "Y":
            is_open_enrollment.append(1)
        else:
            is_open_enrollment.append(0)

    elem_zones["is_open_enrollment"] = is_open_enrollment

    df_districts = (
        elem_zones.groupby(["leaid"], as_index=False)
        .agg({"leaid": "first", "ncessch": "count", "is_open_enrollment": "sum"})
        .rename(
            columns={
                "ncessch": "num_schools",
                "is_open_enrollment": "num_open_enrollment_schools",
            }
        )
    )

    df_districts = pd.DataFrame(df_districts)

    # Next, compute segregation values for each of those districts
    data = []
    for state in os.listdir(school_files_dir):
        data.append(
            pd.read_csv(
                os.path.join(
                    school_files_dir, state, "schools_file_for_assignment.csv"
                ),
                dtype="str",
            )
        )

    df_schools = pd.concat(data)
    df_schools_g = df_schools.groupby(["leaid"], as_index=False).agg(
        {"leaid": "first", "district_totenrl": "first"}
    )

    df_schools_g["district_totenrl"] = df_schools_g["district_totenrl"].astype(float)
    df_districts = pd.merge(
        df_districts, df_schools_g, how="inner", on="leaid"
    ).sort_values(by="district_totenrl", ascending=False)

    keys = {
        "hisp": "hispanic",
        "black": "black",
        "white": "white",
        "frl": "frl",
        "ell": "ell",
        "asian": "asian",
    }

    dissim_indices = {
        "leaid": [],
        "hisp_dissim": [],
        "black_dissim": [],
        "white_dissim": [],
        "frl_dissim": [],
        "ell_dissim": [],
        "asian_dissim": [],
    }

    normalized_exposure_indices = {
        "leaid": [],
        "hisp_norm_exp": [],
        "black_norm_exp": [],
        "white_norm_exp": [],
        "frl_norm_exp": [],
        "ell_norm_exp": [],
        "asian_norm_exp": [],
    }

    gini_indices = {
        "leaid": [],
        "hisp_gini": [],
        "black_gini": [],
        "white_gini": [],
        "frl_gini": [],
        "ell_gini": [],
        "asian_gini": [],
    }

    district_ids = df_districts["leaid"].tolist()
    for dist in district_ids:
        df_curr = df_schools[df_schools["leaid"] == dist].reset_index(drop=True)
        dissim_indices["leaid"].append(dist)
        normalized_exposure_indices["leaid"].append(dist)
        gini_indices["leaid"].append(dist)
        for k in keys:
            try:
                curr_dissim = compute_dissimilarity_index(
                    df_curr, "total_{}".format(keys[k])
                )
            except Exception as e:
                print(e)
                curr_dissim = float("nan")
            dissim_indices["{}_dissim".format(k)].append(curr_dissim)
            try:
                curr_norm_exp = compute_normalized_exposure_index(
                    df_curr, "total_{}".format(keys[k])
                )
            except Exception as e:
                print(e)
                curr_norm_exp = float("nan")
            normalized_exposure_indices["{}_norm_exp".format(k)].append(curr_norm_exp)
            try:
                curr_gini = compute_gini_index(df_curr, "total_{}".format(keys[k]))
            except Exception as e:
                print(e)
                curr_gini = float("nan")
            gini_indices["{}_gini".format(k)].append(curr_gini)

    df_dissim = pd.DataFrame(data=dissim_indices)
    df_norm_exp = pd.DataFrame(data=normalized_exposure_indices)
    df_gini = pd.DataFrame(data=gini_indices)
    df_districts = pd.merge(df_districts, df_dissim, how="inner", on="leaid")
    df_districts = pd.merge(df_districts, df_norm_exp, how="inner", on="leaid")
    df_districts = pd.merge(df_districts, df_gini, how="inner", on="leaid")

    print(df_districts["district_totenrl"].sum())
    df_districts = df_districts[
        (df_districts["num_schools"] > 1) & (df_districts["num_schools"] <= 200)
    ].reset_index(drop=True)
    print(len(df_districts))
    df_districts = df_districts[
        df_districts["num_open_enrollment_schools"] == 0
    ].reset_index(drop=True)

    print(df_districts["district_totenrl"].sum())
    df_districts = df_districts.iloc[N_START:N_END]
    print(df_districts["district_totenrl"].sum())
    df_districts["leaid"] = df_districts["leaid"].str.rjust(7, "0")
    df_districts.to_csv(output_file.format(N_START, N_END), index=False)


def identify_school_pairs_different_demos(
    schools_file="data/derived_data/2122/NC/schools_file_for_assignment.csv",
    district_id=3702970,
    miles_threshold=3,
    output_file="data/school_covariates/cms_divergent_school_pairs_{}_miles.csv",
):

    from utils.distances_and_times import get_distance_for_coord_pair

    df_schools = pd.read_csv(schools_file)
    df_schools = df_schools[df_schools["leaid"] == district_id].reset_index(drop=True)
    dist_perwhite = df_schools["district_perwht"][0]
    print(dist_perwhite)

    df_schools["school_idx"] = pd.Categorical(
        df_schools["ncessch"], categories=df_schools["ncessch"].unique()
    ).codes

    # Identify nearby schools
    dist_mat = [
        [0 for j in range(0, len(df_schools))] for i in range(0, len(df_schools))
    ]
    school_inds = df_schools["school_idx"].tolist()
    nearby_schools = defaultdict(list)
    for i in school_inds:
        school_i = df_schools[df_schools["school_idx"] == i].iloc[0]
        for j in school_inds:
            if i == j:
                continue
            school_j = df_schools[df_schools["school_idx"] == j].iloc[0]
            dist_mat[i][j] = get_distance_for_coord_pair(
                [
                    (school_i["lat"], school_i["long"]),
                    (school_j["lat"], school_j["long"]),
                ]
            )
            if dist_mat[i][j] <= miles_threshold:
                nearby_schools[i].append(j)

    # Now, identify pairs of schools that are nearby that diverge in opposite directions from the district-wide percentage_white threshold
    divergent_pairs = []
    for s1 in nearby_schools:
        school_1 = df_schools[df_schools["school_idx"] == s1].iloc[0]
        for s2 in nearby_schools[s1]:
            school_2 = df_schools[df_schools["school_idx"] == s2].iloc[0]
            if (
                school_1["perwht"] > dist_perwhite
                and school_2["perwht"] < dist_perwhite
            ) or (
                school_1["perwht"] < dist_perwhite
                and school_2["perwht"] > dist_perwhite
            ):

                abs_diff = np.abs(school_1["perwht"] - school_2["perwht"])
                pair = (school_1["school_idx"], school_2["school_idx"], abs_diff)
                pair_reverse = (
                    school_2["school_idx"],
                    school_1["school_idx"],
                    abs_diff,
                )
                if not pair_reverse in divergent_pairs:
                    divergent_pairs.append(pair)

    # Create output data frame
    pairs_data = {
        "school_1_name": [],
        "school_1_nces": [],
        "school_1_perwht": [],
        "school_2_name": [],
        "school_2_nces": [],
        "school_2_perwht": [],
        "abs_diff_in_per_wht": [],
        "distance_in_miles": [],
    }
    for p in divergent_pairs:
        school_1 = df_schools[df_schools["school_idx"] == p[0]].iloc[0]
        school_2 = df_schools[df_schools["school_idx"] == p[1]].iloc[0]
        pairs_data["school_1_name"].append(school_1["SCH_NAME"])
        pairs_data["school_1_nces"].append(school_1["ncessch"])
        pairs_data["school_1_perwht"].append(school_1["perwht"])
        pairs_data["school_2_name"].append(school_2["SCH_NAME"])
        pairs_data["school_2_nces"].append(school_2["ncessch"])
        pairs_data["school_2_perwht"].append(school_2["perwht"])
        pairs_data["abs_diff_in_per_wht"].append(p[2])
        pairs_data["distance_in_miles"].append(dist_mat[p[0]][p[1]])

    pd.DataFrame(data=pairs_data).to_csv(
        output_file.format(miles_threshold), index=False
    )


def identify_nearby_cms_magnet_voices_families(
    cms_families_file="data/school_covariates/052722_contactInfo_with_region.csv",
    schools_file="data/school_covariates/cms_divergent_school_pairs_3_miles.csv",
    full_blocks_file="data/derived_data/2122/NC/blocks_file_for_assignment.csv",
    output_file="{}_nearby_families.csv",
):

    df_cms = pd.read_csv(cms_families_file, dtype=str)
    df_cms["region_id"] = df_cms["region_id"].str.strip(".0")
    df_s = pd.read_csv(schools_file, dtype=str)
    nces_id_to_name = {}
    for i in range(0, len(df_s)):
        nces_id_to_name[df_s["school_1_nces"][i]] = df_s["school_1_name"][i]
        nces_id_to_name[df_s["school_2_nces"][i]] = df_s["school_2_name"][i]
    df_b = pd.read_csv(full_blocks_file, dtype=str)
    df_b["tract_id"] = df_b["block_id"].str[:11]

    potential_family_contacts = {
        "speaker_id": [],
        "nearby_school_nces": [],
        "nearby_school_name": [],
    }

    for s in nces_id_to_name:
        df_blocks_curr = df_b[df_b["ncessch"] == s].reset_index(drop=True)
        curr_tract_ids = list(set(df_blocks_curr["tract_id"].tolist()))
        for t in curr_tract_ids:
            df_cms_curr = df_cms[df_cms["region_id"] == t].reset_index(drop=True)
            for i in range(0, len(df_cms_curr)):
                potential_family_contacts["speaker_id"].append(
                    df_cms_curr["speaker_id"][i]
                )
                potential_family_contacts["nearby_school_nces"].append(s)
                potential_family_contacts["nearby_school_name"].append(
                    nces_id_to_name[s]
                )

    df_fam = pd.DataFrame(data=potential_family_contacts)
    df_fam.to_csv(output_file.format(schools_file.split(".csv")[0]), index=False)


def analyze_choice_options_in_districts(
    sab_elem_file="data/school_boundary_shapefiles/ATTOM_boundaries_21_22_USA/school-attendance-areas-03.shp",
    raw_school_files_dir="data/school_covariates/ccd_1920_school_covars_by_state/",
    school_files_dir="data/derived_data/2122",
    charter_schools_file="data/school_covariates/ccd_19_20_charter_status.csv",
    magnet_schools_file="data/school_covariates/ccd_19_20_magnet_status.csv",
    schools_locations_file="data/school_covariates/EDGE_GEOCODE_PUBLICSCH_1920.csv",
    simulated_dist_results="simulation_outputs/2122_top_100_dissim_longer/consolidated_simulation_results.csv",
    top_dists_file="data/school_covariates/top_100_largest_districts.csv",
    output_file="data/school_covariates/school_choice_analysis.csv",
):
    # First, identify lat / long of all charter and magnet elem schools (since we only include non open erollment schools, don't expect magnets?)
    # Next, identify which ones are contained in which of our 98 school districts
    # Next, determine ratio of students at these schools to the in-dist elem options (to determine approx. rates of choice)
    # Next, determine how their racial breakdown compares to those of the district as a whole (in terms of white/non-white)
    # Use this to inform choice scenarios — what if the ratio grew by X%?  Y%?  **need to do this by school, different schools prob have diff opt out rates

    df_geos = pd.read_csv(schools_locations_file, dtype=str)[["NCESSCH", "LAT", "LON"]]
    df_public_universe = pd.read_csv(
        charter_schools_file, encoding="ISO-8859-1", dtype=str
    )
    df_public_universe_elem = df_public_universe[
        df_public_universe["LEVEL"] == "Elementary"
    ].reset_index(drop=True)
    df_charters = df_public_universe_elem[
        df_public_universe_elem["CHARTER_TEXT"] == "Yes"
    ].reset_index(drop=True)[["LEAID", "NCESSCH", "CHARTER_TEXT"]]
    df_magnets = pd.read_csv(magnet_schools_file, encoding="ISO-8859-1", dtype=str)
    df_magnets = df_magnets[
        (df_magnets["MAGNET_TEXT"] == "Yes")
        & (df_magnets["NCESSCH"].isin(df_public_universe_elem["NCESSCH"]))
    ].reset_index(drop=True)[["LEAID", "NCESSCH", "MAGNET_TEXT"]]
    df_charters["LEAID"] = add_leading_zero_for_district(df_charters["LEAID"])
    df_charters["NCESSCH"] = add_leading_zero_for_school(df_charters["NCESSCH"])
    df_magnets["LEAID"] = add_leading_zero_for_district(df_magnets["LEAID"])
    df_magnets["NCESSCH"] = add_leading_zero_for_school(df_magnets["NCESSCH"])
    df_choice_schools = pd.merge(df_charters, df_magnets, on="NCESSCH", how="outer")

    # Merge lat long
    df_choice_schools = pd.merge(df_choice_schools, df_geos, on="NCESSCH", how="inner")

    # Next, get the attendance boundaries of all of our 98 districts
    df_soln = pd.read_csv(simulated_dist_results, dtype=str)
    curr_districts = df_soln["district_id"].unique()
    print("Loading elementary school zones")
    elem_zones = (
        gpd.read_file(sab_elem_file)
        .to_crs(epsg=4326)
        .rename(
            columns={
                "NCESDISTID": "leaid",
                "NCESSCHID": "ncessch",
                "OPENENROLL": "openEnroll",
                "AREASQMI": "Shape_Area",
            }
        )
    )
    is_open_enrollment = []
    for i in range(0, len(elem_zones)):
        if elem_zones["openEnroll"][i] == "Y":
            is_open_enrollment.append(1)
        else:
            is_open_enrollment.append(0)

    elem_zones["is_open_enrollment"] = is_open_enrollment
    elem_zones = elem_zones[elem_zones["leaid"].isin(curr_districts)].reset_index(
        drop=True
    )

    # Now, let's get identify which district boundaries these choice schools are in
    df_choice_schools = gpd.GeoDataFrame(df_choice_schools)
    df_choice_schools["geometry"] = gpd.points_from_xy(
        df_choice_schools.LON, df_choice_schools.LAT
    )
    df_choice_schools = df_choice_schools.set_geometry("geometry")
    choice_schools_in_zones = gpd.sjoin(
        df_choice_schools, elem_zones, how="inner", op="intersects"
    ).reset_index(drop=True)

    # Now, let's merge on enrollment data for these schools
    data = []
    for state in os.listdir(raw_school_files_dir):
        data.append(
            pd.read_csv(
                os.path.join(
                    raw_school_files_dir, state, "ccd_1920_crdc_1718_school_covars.csv"
                ),
                dtype="str",
            )
        )

    df_schools = pd.concat(data)

    df_schools["total_students"] = df_schools["total_students"].astype(float)
    df_schools["total_white"] = df_schools["num_white"].astype(float)
    df_schools["total_native"] = df_schools["num_native"].astype(float)
    df_schools["total_asian"] = df_schools["num_asian"].astype(float)
    df_schools["total_hispanic"] = df_schools["num_hispanic"].astype(float)
    df_schools["total_black"] = df_schools["num_black"].astype(float)

    choice_schools_in_zones = pd.merge(
        choice_schools_in_zones,
        df_schools,
        on="NCESSCH",
        how="inner",
    )
    print(len(choice_schools_in_zones))

    df_top_dists = pd.read_csv(top_dists_file, dtype={"leaid": str})
    df_top_dists = df_top_dists[df_top_dists["leaid"].isin(curr_districts)][
        ["leaid", "num_schools", "district_totenrl"]
    ]

    data = []
    for state in os.listdir(school_files_dir):
        data.append(
            pd.read_csv(
                os.path.join(
                    school_files_dir, state, "schools_file_for_assignment.csv"
                ),
                dtype={"leaid": str, "ncessch": str},
            )
        )

    df_schools = pd.concat(data)
    df_schools = df_schools[df_schools["leaid"].isin(curr_districts)]
    df_schools_g = (
        df_schools.groupby(["leaid"], as_index=False)
        .agg(
            {
                "leaid": "first",
                "total_enrollment": "sum",
                "total_white": "sum",
                "total_native": "sum",
                "total_asian": "sum",
                "total_hispanic": "sum",
                "total_black": "sum",
            }
        )
        .rename(
            columns={
                "total_enrollment": "dist_total_students",
                "total_white": "dist_total_white",
                "total_native": "dist_total_native",
                "total_asian": "dist_total_asian",
                "total_hispanic": "dist_total_hispanic",
                "total_black": "dist_total_black",
            }
        )
    )

    choice_schools_by_dist_loc = (
        choice_schools_in_zones.groupby(["leaid"], as_index=False)
        .agg(
            {
                "leaid": "first",
                "ncessch": "count",
                "total_students": "sum",
                "total_white": "sum",
                "total_native": "sum",
                "total_asian": "sum",
                "total_hispanic": "sum",
                "total_black": "sum",
            }
        )
        .rename(
            columns={
                "ncessch": "num_charter_or_magnet_schools",
                "total_students": "c_m_total_students",
                "total_white": "c_m_total_white",
                "total_native": "c_m_total_native",
                "total_asian": "c_m_total_asian",
                "total_hispanic": "c_m_total_hispanic",
                "total_black": "c_m_total_black",
            }
        )
    )

    df_merged = pd.merge(
        df_top_dists, choice_schools_by_dist_loc, on="leaid", how="inner"
    )
    print(len(df_merged))

    df_merged = pd.merge(df_merged, df_schools_g, on="leaid", how="inner")
    print(len(df_merged))

    df_merged["ratio_c_or_m_to_dist_enroll"] = (
        df_merged["c_m_total_students"] / df_merged["dist_total_students"]
    )

    df_merged["ratio_c_or_m_to_dist_white"] = (
        df_merged["c_m_total_white"] / df_merged["dist_total_white"]
    )

    df_merged["ratio_c_or_m_to_dist_non_white"] = (
        df_merged["c_m_total_students"] - df_merged["c_m_total_white"]
    ) / (df_merged["dist_total_students"] - df_merged["dist_total_white"])

    df_merged["ratio_c_or_m_to_dist_black"] = (
        df_merged["c_m_total_black"] / df_merged["dist_total_black"]
    )

    df_merged["ratio_c_or_m_to_dist_native"] = (
        df_merged["c_m_total_native"] / df_merged["dist_total_native"]
    )

    df_merged["ratio_c_or_m_to_dist_hispanic"] = (
        df_merged["c_m_total_hispanic"] / df_merged["dist_total_hispanic"]
    )

    df_merged["ratio_c_or_m_to_dist_asian"] = (
        df_merged["c_m_total_asian"] / df_merged["dist_total_asian"]
    )

    print(
        df_merged["ratio_c_or_m_to_dist_enroll"].median(),
        df_merged["ratio_c_or_m_to_dist_enroll"].mean(),
        df_merged["ratio_c_or_m_to_dist_enroll"].max(),
    )
    print(
        df_merged["ratio_c_or_m_to_dist_white"].median(),
        df_merged["ratio_c_or_m_to_dist_white"].mean(),
        df_merged["ratio_c_or_m_to_dist_white"].max(),
    )

    print(
        df_merged["ratio_c_or_m_to_dist_non_white"].median(),
        df_merged["ratio_c_or_m_to_dist_non_white"].mean(),
        df_merged["ratio_c_or_m_to_dist_non_white"].max(),
    )
    df_merged.to_csv(output_file, index=False)


def analyze_adjacent_districts(
    schools_file="data/derived_data/2122/GA/schools_file_for_assignment.csv",
    adjacent_dists_file="data/school_district_2021_boundaries/district_neighbors.json",
    selected_dist="1300120",
):
    df = pd.read_csv(schools_file, dtype={"leaid": str})
    df_g = df.groupby("leaid", as_index=False).agg(
        {
            "leaid": "first",
            "district_totenrl": "first",
            "district_perwht": "first",
            "district_perblk": "first",
            "district_perhsp": "first",
            "district_perasn": "first",
            "district_pernam": "first",
        }
    )
    dist_neighbors = read_dict(adjacent_dists_file)
    neighbors_of_curr_dist = dist_neighbors[selected_dist]
    df_g = df_g[df_g["leaid"].isin(neighbors_of_curr_dist)].reset_index(drop=True)
    print(df_g)


if __name__ == "__main__":
    # analyze_historical_rezonings()
    # identify_largest_districts_with_all_closed_enrollment()
    # count_fraction_closed_enrollment_districts()
    # identify_school_pairs_different_demos()
    # identify_nearby_cms_magnet_voices_families()
    # analyze_choice_options_in_districts()
    analyze_adjacent_districts()
