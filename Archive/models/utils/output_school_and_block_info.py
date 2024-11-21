from utils.header import *
from utils.segregation import compute_school_segregation, allocate_students_to_blocks


def output_school_file_for_assignment(
    input_dir="data/derived_data/{}/",
    input_file="data/derived_data/{}/{}/full_blocks_data.csv",
    schools_info_file="data/school_covariates/school_and_district_info_ccd_1920.csv",
    output_file="data/derived_data/{}/{}/schools_file_for_assignment.csv",
    zones_years=["2122"],
):

    for y in zones_years:
        curr_states = os.listdir(input_dir.format(y))
        # curr_states = ["VA"]
        df_schools_districts_metadata = pd.read_csv(
            schools_info_file, encoding="ISO-8859-1", dtype=str
        )
        df_schools_metadata = df_schools_districts_metadata.groupby(
            "NCESSCH", as_index=False
        ).agg({"SCH_NAME": "first", "LEA_NAME": "first"})

        for state in curr_states:
            print(y, state)

            # Aggregate school-level values
            df = pd.read_csv(
                input_file.format(y, state), encoding="ISO-8859-1", dtype=str
            ).fillna(0)
            df_g = (
                df.groupby(["ncessch"])
                .agg(
                    {
                        "block_id": "count",
                        "leaid": "first",
                        "totenrl": "first",
                        "perwht": "first",
                        "pernam": "first",
                        "perasn": "first",
                        "perhsp": "first",
                        "perblk": "first",
                        "perfrl": "first",
                        "perell": "first",
                        "openEnroll": "first",
                        "lat": "first",
                        "long": "first",
                    }
                )
                .rename(
                    columns={"block_id": "num_blocks", "totenrl": "total_enrollment"}
                )
                .reset_index()
            )

            df_g = pd.merge(
                df_g,
                df_schools_metadata,
                left_on="ncessch",
                right_on="NCESSCH",
                how="inner",
            )

            # Convert relevant columns to numeric
            df_g["total_enrollment"] = pd.to_numeric(df_g["total_enrollment"])
            df_g["perwht"] = pd.to_numeric(df_g["perwht"])
            df_g["pernam"] = pd.to_numeric(df_g["pernam"])
            df_g["perasn"] = pd.to_numeric(df_g["perasn"])
            df_g["perhsp"] = pd.to_numeric(df_g["perhsp"])
            df_g["perblk"] = pd.to_numeric(df_g["perblk"])
            df_g["perfrl"] = pd.to_numeric(df_g["perfrl"])
            df_g["perell"] = pd.to_numeric(df_g["perell"])

            df_g["total_enrollment"] = np.maximum(np.round(df_g["total_enrollment"]), 0)
            df_g["total_white"] = np.maximum(
                df_g["total_enrollment"] * df_g["perwht"], 0
            )
            df_g["total_native"] = np.maximum(
                df_g["total_enrollment"] * df_g["pernam"], 0
            )
            df_g["total_asian"] = np.maximum(
                df_g["total_enrollment"] * df_g["perasn"], 0
            )
            df_g["total_hispanic"] = np.maximum(
                df_g["total_enrollment"] * df_g["perhsp"], 0
            )
            df_g["total_black"] = np.maximum(
                df_g["total_enrollment"] * df_g["perblk"], 0
            )
            df_g["total_frl"] = np.maximum(df_g["total_enrollment"] * df_g["perfrl"], 0)
            df_g["total_ell"] = np.maximum(df_g["total_enrollment"] * df_g["perell"], 0)

            # # Filter out schools with 0 total enrollment
            df_g = df_g[df_g["total_enrollment"] > 0].reset_index()

            # Compute district-level values
            df_district = (
                df_g.groupby(["leaid"])
                .agg(
                    {
                        "ncessch": "count",
                        "total_enrollment": "sum",
                        "total_white": "sum",
                        "total_native": "sum",
                        "total_asian": "sum",
                        "total_hispanic": "sum",
                        "total_black": "sum",
                        "total_frl": "sum",
                        "total_ell": "sum",
                    }
                )
                .rename(columns={"ncessch": "num_schools"})
                .reset_index()
                .sort_values(by="total_enrollment", ascending=False)
            )

            df_district["perwht"] = (
                df_district["total_white"] / df_district["total_enrollment"]
            )
            df_district["pernam"] = (
                df_district["total_native"] / df_district["total_enrollment"]
            )
            df_district["perasn"] = (
                df_district["total_asian"] / df_district["total_enrollment"]
            )
            df_district["perhsp"] = (
                df_district["total_hispanic"] / df_district["total_enrollment"]
            )
            df_district["perblk"] = (
                df_district["total_black"] / df_district["total_enrollment"]
            )
            df_district["perfrl"] = (
                df_district["total_frl"] / df_district["total_enrollment"]
            )
            df_district["perell"] = (
                df_district["total_ell"] / df_district["total_enrollment"]
            )

            data_to_merge = {
                "ncessch": [],
                "seg_index_perblk": [],
                "seg_index_pernam": [],
                "seg_index_perasn": [],
                "seg_index_perhsp": [],
                "seg_index_perfrl": [],
                "seg_index_perell": [],
                "seg_index_perwht": [],
                "district_totenrl": [],
                "district_perwht": [],
                "district_perblk": [],
                "district_pernam": [],
                "district_perasn": [],
                "district_perhsp": [],
                "district_perfrl": [],
                "district_perell": [],
                "district_num_schools": [],
            }

            # Compute segregation values per school
            for i in range(0, len(df_g)):

                school = df_g.iloc[i]
                district = df_district[df_district["leaid"].isin([school["leaid"]])]
                data_to_merge["ncessch"].append(school["ncessch"])
                data_to_merge["district_totenrl"].append(
                    float(district["total_enrollment"])
                )
                data_to_merge["district_perwht"].append(float(district["perwht"]))
                data_to_merge["district_perblk"].append(float(district["perblk"]))
                data_to_merge["district_pernam"].append(float(district["pernam"]))
                data_to_merge["district_perhsp"].append(float(district["perhsp"]))
                data_to_merge["district_perasn"].append(float(district["perasn"]))
                data_to_merge["district_perfrl"].append(float(district["perfrl"]))
                data_to_merge["district_perell"].append(float(district["perell"]))
                data_to_merge["district_num_schools"].append(
                    float(district["num_schools"])
                )

                seg_indices = compute_school_segregation(school, district)
                data_to_merge["seg_index_perblk"].append(seg_indices["perblk"])
                data_to_merge["seg_index_pernam"].append(seg_indices["pernam"])
                data_to_merge["seg_index_perasn"].append(seg_indices["perasn"])
                data_to_merge["seg_index_perhsp"].append(seg_indices["perhsp"])
                data_to_merge["seg_index_perfrl"].append(seg_indices["perfrl"])
                data_to_merge["seg_index_perell"].append(seg_indices["perell"])
                data_to_merge["seg_index_perwht"].append(seg_indices["perwht"])

            df_merge = pd.DataFrame(data=data_to_merge)
            df_g = pd.merge(df_g, df_merge, how="left", on="ncessch").reset_index()
            # df_g = df_g.drop(columns=["level_0", "index"])
            df_g = df_g.drop(columns=["index"])
            df_g.to_csv(output_file.format(y, state), index=False)


def output_block_file_for_assignment(
    state,
    y,
    input_file="data/derived_data/{}/{}/full_blocks_data.csv",
    input_schools_file="data/derived_data/{}/{}/schools_file_for_assignment.csv",
    output_file="data/derived_data/{}/{}/blocks_file_for_assignment.csv",
    race_cat_keys={
        "perwht": "num_white_to_school",
        "perblk": "num_black_to_school",
        "perasn": "num_asian_to_school",
        "pernam": "num_native_to_school",
    },
    hisp_cat_keys={
        "perhsp": "num_hispanic_to_school",
    },
    ell_cat_keys={
        "perell": "num_ell_to_school",
    },
    frl_cat_keys={"perfrl": "num_frl_to_school"},
    total_cat_keys={"pertotal": "num_total_to_school"},
):

    print(state)
    df_blocks = pd.read_csv(
        input_file.format(y, state), encoding="ISO-8859-1", dtype=str
    ).fillna(0)
    df_schools = pd.read_csv(
        input_schools_file.format(y, state), encoding="ISO-8859-1", dtype=str
    ).fillna(0)

    # Add this in to make the allocation code easier / more general
    df_schools["pertotal"] = [1 for j in range(0, len(df_schools))]

    block_students_by_cat = {}

    for i in range(0, len(df_schools)):
        curr_school = df_schools.iloc[i]

        curr_blocks = df_blocks[
            df_blocks["ncessch"].isin([curr_school["ncessch"]])
        ].reset_index(drop=True)

        # Testing
        # curr_school = df_schools[
        #     df_schools["ncessch"].isin(["510126000483"])
        # ].reset_index(drop=True)
        # curr_blocks = df_blocks[
        #     df_blocks["ncessch"].isin(["510126000483"])
        # ].reset_index(drop=True)

        all_cat_keys = [
            race_cat_keys,
            hisp_cat_keys,
            ell_cat_keys,
            frl_cat_keys,
            total_cat_keys,
        ]
        blocks_for_curr_school = allocate_students_to_blocks(
            curr_school, curr_blocks, all_cat_keys
        )
        block_students_by_cat.update(blocks_for_curr_school)

        # Check to make sure the sum of students per category from each block is always <= the school's total enrollment
        for keypair in all_cat_keys:
            for val in keypair.values():
                cat_total_to_school = 0
                school_total = 0
                for b in blocks_for_curr_school:
                    cat_total_to_school += blocks_for_curr_school[b][val]
                    school_total += blocks_for_curr_school[b]["num_total_to_school"]
                assert cat_total_to_school <= school_total

    # Initialize dict and create dataframe
    data_allocations = {}
    for cats in all_cat_keys:
        data_allocations.update({k: [] for k in list(cats.values())})
    data_allocations["block_id"] = []

    for b in block_students_by_cat:
        data_allocations["block_id"].append(b)
        for cats in all_cat_keys:
            for k in list(cats.values()):
                data_allocations[k].append(block_students_by_cat[b][k])

    df_allocations = pd.DataFrame(data=data_allocations)
    df_blocks = df_blocks[
        df_blocks["block_id"].isin(list(block_students_by_cat.keys()))
    ].reset_index(drop=True)
    df_blocks = pd.merge(df_blocks, df_allocations, how="left", on="block_id")

    keys_to_keep = []
    for cats in all_cat_keys:
        keys_to_keep.extend(deepcopy(list(cats.values())))

    keys_to_keep.extend(
        [
            "block_id",
            "block_centroid_lat",
            "block_centroid_long",
            "ncessch",
            "distance_to_school",
            "travel_time_to_school",
            "directions_to_school",
        ]
    )
    df_blocks = df_blocks[keys_to_keep]
    df_blocks.to_csv(output_file.format(y, state), index=False)


def output_block_file_for_assignment_parallel(
    input_dir="data/derived_data/{}/",
    zones_years=["2122"],
    processing_function=output_block_file_for_assignment,
):

    N_THREADS = 10

    all_states_to_process = []
    for y in zones_years:
        for state in os.listdir(input_dir.format(y)):
            all_states_to_process.append((state, y))

    # all_states_to_process = [("SD", "2122")]
    # processing_function(*all_states_to_process[0])
    print("Starting multiprocessing ...")
    print(len(all_states_to_process))
    from multiprocessing import Pool

    p = Pool(N_THREADS)
    p.starmap(processing_function, all_states_to_process)

    p.terminate()
    p.join()


if __name__ == "__main__":
    # output_school_file_for_assignment()
    output_block_file_for_assignment_parallel()
