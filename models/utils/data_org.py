from utils.header import *


def unzip_all_block_files(block_shapes_dir="data/census_block_shapefiles_2020/"):

    import zipfile

    all_files_and_folders = os.listdir(block_shapes_dir)
    all_zip_files_to_extract = [
        f
        for f in all_files_and_folders
        if f.endswith(".zip") and not f.split(".zip")[0] in all_files_and_folders
    ]
    for f in all_zip_files_to_extract:
        print(f)
        with zipfile.ZipFile(block_shapes_dir + f, "r") as zip_ref:
            dest_dir = f.split(".zip")[0]
            zip_ref.extractall(block_shapes_dir + dest_dir)


def output_block_racial_demos_by_state(
    states_codes_file="data/state_codes.csv",
    block_demos_file="data/census_block_covariates/2020_census_race_hisp_over_18_by_block/filtered_demos.csv",
    output_dir="data/census_block_covariates/2020_census_race_hisp_over_18_by_block/by_state/",
    output_file_name="racial_demos_by_block.csv",
):

    df_states = pd.read_csv(states_codes_file, dtype="str")
    # Only process those states we haven't processed yet
    # states_already_computed = [f for f in os.listdir(output_dir)]
    # df_states = df_states[
    #     ~df_states["abbrev"].isin(states_already_computed)
    # ].reset_index()

    print("Loading filtered demos data ...")
    df_demos = pd.read_csv(block_demos_file, encoding="ISO-8859-1", dtype="str")
    print("Iterating over states ...")
    for i in range(0, len(df_states)):
        try:
            state_abbrev = df_states["abbrev"][i]
            state_code = str(df_states["fips_code"][i])
            if len(state_code) < 2:
                state_code = "0" + state_code
            curr_demos = df_demos[df_demos["STATEA"].isin([state_code])].reset_index(
                drop=True
            )
            print(state_abbrev, state_code, len(curr_demos))

            curr_output_dir = output_dir + state_abbrev + "/"
            if not os.path.exists(curr_output_dir):
                os.mkdir(curr_output_dir)
            curr_demos.to_csv(curr_output_dir + output_file_name, index=False)
        except Exception as e:
            print(e)


def output_school_covars_by_state(
    states_codes_file="data/state_codes.csv",
    school_race_covars_file="data/school_covariates/ccd_race_1920.csv",
    school_frl_covars_file="data/school_covariates/ccd_frl_1920.csv",
    school_ell_covars_file="data/school_covariates/crdc_1718/files/Data/SCH/CRDC/CSV/Enrollment.csv",
    output_dir="data/school_covariates/ccd_1920_school_covars_by_state/",
    output_file_name="ccd_1920_crdc_1718_school_covars.csv",
):

    df_states = pd.read_csv(states_codes_file, dtype="str")
    # Only process those states we haven't processed yet
    # states_already_computed = [f for f in os.listdir(output_dir)]
    # df_states = df_states[
    #     ~df_states["abbrev"].isin(states_already_computed)
    # ].reset_index()

    print("Loading school race covars data ...")
    df_race_covars = pd.read_csv(
        school_race_covars_file, encoding="ISO-8859-1", dtype="str"
    )

    print("Loading school frl covars data ...")
    df_frl_covars = pd.read_csv(
        school_frl_covars_file, encoding="ISO-8859-1", dtype="str"
    )

    print("Loading school ell covars data ...")
    df_ell_covars = pd.read_csv(
        school_ell_covars_file, encoding="ISO-8859-1", dtype="str"
    )

    df_race_covars["STUDENT_COUNT"] = df_race_covars["STUDENT_COUNT"].astype(float)
    df_frl_covars["STUDENT_COUNT"] = df_frl_covars["STUDENT_COUNT"].astype(float)
    df_ell_covars["SCH_ENR_LEP_M"] = df_ell_covars["SCH_ENR_LEP_M"].astype(float)
    df_ell_covars["SCH_ENR_LEP_F"] = df_ell_covars["SCH_ENR_LEP_F"].astype(float)

    race_keys = {
        "White": "num_white",
        "Black or African American": "num_black",
        "Hispanic/Latino": "num_hispanic",
        "American Indian or Alaska Native": "num_native",
        "Asian": "num_asian",
    }

    print("Iterating over states ...")
    for i in range(0, len(df_states)):
        try:
            state_abbrev = df_states["abbrev"][i]
            # if not state_abbrev == "VA":
            #     continue

            # RACE AND ETHNICITY
            curr_race_covars = df_race_covars[
                df_race_covars["ST"].isin([state_abbrev])
            ].reset_index(drop=True)
            curr_race_covars = curr_race_covars[
                ["NCESSCH", "RACE_ETHNICITY", "STUDENT_COUNT", "TOTAL_INDICATOR"]
            ]
            print(state_abbrev, len(curr_race_covars))

            df_s = (
                curr_race_covars[
                    curr_race_covars["TOTAL_INDICATOR"].isin(
                        ["Derived - Education Unit Total minus Adult Education Count"]
                    )
                ][["NCESSCH", "STUDENT_COUNT"]]
                .rename(columns={"STUDENT_COUNT": "total_students"})
                .reset_index(drop=True)
            )

            for r in race_keys:
                filtered_df = curr_race_covars[
                    curr_race_covars["RACE_ETHNICITY"].isin([r])
                    & curr_race_covars["TOTAL_INDICATOR"].isin(
                        [
                            "Derived - Subtotal by Race/Ethnicity and Sex minus Adult Education Count"
                        ]
                    )
                ]
                df_curr = (
                    filtered_df.groupby(["NCESSCH"])
                    .agg(
                        {
                            "STUDENT_COUNT": "sum",
                        }
                    )
                    .rename(columns={"STUDENT_COUNT": race_keys[r]})
                )
                df_s = pd.merge(df_s, df_curr, on="NCESSCH", how="left")

            # FRL
            curr_frl_covars = df_frl_covars[
                df_frl_covars["ST"].isin([state_abbrev])
            ].reset_index(drop=True)
            curr_frl_covars = curr_frl_covars[
                ["NCESSCH", "LUNCH_PROGRAM", "STUDENT_COUNT"]
            ]
            filtered_df = curr_frl_covars[
                curr_frl_covars["LUNCH_PROGRAM"].isin(
                    ["Free lunch qualified", "Reduced-price lunch qualified"]
                )
            ]
            df_frl = (
                filtered_df.groupby(["NCESSCH"])
                .agg({"STUDENT_COUNT": "sum"})
                .rename(columns={"STUDENT_COUNT": "num_frl"})
            )
            df_s = pd.merge(df_s, df_frl, on="NCESSCH", how="left")

            # ELL
            curr_ell_covars = df_ell_covars[
                df_ell_covars["LEA_STATE"].isin([state_abbrev])
            ].reset_index(drop=True)
            curr_ell_covars = curr_ell_covars[
                ["COMBOKEY", "SCH_ENR_LEP_M", "SCH_ENR_LEP_F"]
            ]
            curr_ell_covars["num_ell"] = (
                curr_ell_covars["SCH_ENR_LEP_M"] + curr_ell_covars["SCH_ENR_LEP_F"]
            )

            # all_nces = set(df_s["NCESSCH"].tolist())
            # ell_nces = set(curr_ell_covars["COMBOKEY"].tolist())
            # print(len(all_nces.intersection(ell_nces)) / len(all_nces.union(ell_nces)))
            # exit()
            df_s = pd.merge(
                df_s,
                curr_ell_covars,
                left_on="NCESSCH",
                right_on="COMBOKEY",
                how="left",
            ).drop(columns=["SCH_ENR_LEP_M", "SCH_ENR_LEP_F", "COMBOKEY"])

            curr_output_dir = output_dir + state_abbrev + "/"
            if not os.path.exists(curr_output_dir):
                os.mkdir(curr_output_dir)
            df_s.to_csv(curr_output_dir + output_file_name, index=False)
        except Exception as e:
            print(e)


def output_block_group_demos_by_state(
    states_codes_file="data/state_codes.csv",
    bgrp_file="data/census_block_covariates/2015_2019_ACS_block_group_poverty_ratios/nhgis0014_ds244_20195_blck_grp.csv",
    output_dir="data/census_block_covariates/2015_2019_ACS_block_group_poverty_ratios/by_state/",
    output_file_name="frl_demos_by_block_group.csv",
):

    df_states = pd.read_csv(states_codes_file, dtype="str")
    # Only process those states we haven't processed yet
    states_already_computed = [f for f in os.listdir(output_dir)]
    df_states = df_states[
        ~df_states["abbrev"].isin(states_already_computed)
    ].reset_index()

    print("Loading demos data ...")
    df_demos = pd.read_csv(bgrp_file, encoding="ISO-8859-1", dtype="str")
    for i in range(0, len(df_states)):
        try:
            state_abbrev = df_states["abbrev"][i]
            state_code = str(df_states["fips_code"][i])
            if len(state_code) < 2:
                state_code = "0" + state_code
            curr_demos = df_demos[df_demos["STATEA"].isin([state_code])].reset_index(
                drop=True
            )
            print(state_abbrev, state_code, len(curr_demos))

            curr_output_dir = output_dir + state_abbrev + "/"
            if not os.path.exists(curr_output_dir):
                os.mkdir(curr_output_dir)
            curr_demos.to_csv(curr_output_dir + output_file_name, index=False)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # unzip_all_block_files()
    # output_block_racial_demos_by_state()
    # output_block_group_demos_by_state()
    output_school_covars_by_state()
