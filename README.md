# Repository for "Redrawing attendance boundaries to promote racial and ethnic diversity in elementary schools"

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-xie2018" class="csl-entry">

**Citation:**
Nabeel Gillani, Doug Beeferman, Christine Vega-Pourheydarian, Cassandra Overney, Pascal Van Hentenryck, Deb Roy. *Redrawing attendance boundaries to promote racial and ethnic diversity in elementary schools*. Forthcoming in *Educational Researcher*.
<https://www.ssrn.com/abstract=4387899>.

</div>

</div>

The notes below describe the key code and data files used in / produced from this study.  They likely miss some details and context that might be helpful for replicating and/or building off of this work.  If you find that's the case, please do not hesitate to reach out to <n.gillani@northeastern.edu> with any questions or comments!

# Matt's Notes
I moved several files into `likely_unnecessary_scripts`, including the existing one that drew the census block map. Unfortunately, the data they used there is not provided to us because they had to purchase access to it.

## Downloading solver_files
Since we're only doing Maryland, download `solver_files.zip` from our google drive [here](https://drive.google.com/file/d/1rmAf5k1CNSTL_ghatMiArkBmt_XYuhuF/view?usp=drive_link). Place it in `models/models/` repository next to results.

## Set Up
I recommend the following steps for set up:
1. `conda create -n attendance`
2. `conda activate attendance` (if not already activated)
3. `pip install -r "requirements.txt"`

## Running assignment_cp_sat.py

This is the main file for running the simulation. It takes in data from `models/data/derived_data/.../blocks_file_for_assignment` and `models/models/solver_files/.../` (blocks, prepped, travel time matrix). We can use `simulation_sweeps.py` to run the simulation across different parameters. Run it via `python models/assignment_cp_sat.py` and see the output via `models/models/results/.../solution_[parameters]`.

TODO: Someone might want to take at simulation_sweeps to see if we can adapt/use that code to run the different configurations and school districts in Maryland.

## Evaluating the code with `analyze_model_solution.py`

You can then run `analyze_model_solution.py` to create csvs that detail the actual changes. These can be used to visualize the actual changes

## Running analyze_rezoning_sim_outputs.Rmd

The above Rmd needs serious changing to get to run. I've written some functions to install the necessary packages (which should now work for everyone, some notes about potential `brew` packages that may also need to be installed for those on Mac, specifically Xquartz and Freetype). Someone should take a deeper look at this to see if it's necessary or if we can/should just use Sasha's code for visualization.


# Code

Python package dependencies can be found in `requirements.txt`.  The key code folders/files are as follows:

* `utils/` -> contains code to run the data pipeline (see `data_prep_pipeline.py` in this folder).  Note: travel time matrices were computed on an external server due to large memory requirements.  Files produced by this pipeline can be found in the data folders named `data/derived_data` and `models/solver_files`(more info on these and other data folders found below)
* `models/` -> contains code for the rezoning algorithm (see `assignment_cp_sat.py`).  `simulation_sweeps.py` enables the creation of different parameter configurations to run through the optimization model in the former file (e.g., using a parallel computing cluster)
* `analysis/` -> contains code for sifting through outputs from different simulation configurations/runs and computing key outcome measures.  The file `analyze_model_solution.py` does this and outputs a set of CSVs per district, which are then read in by `analyze_rezoning_sim_outputs.Rmd` for further processing and analysis.  This R markdown file is also responsible for generating the main plots in the paper.  
* `frontend/` -> contains code for the dashboard hosted at <https://schooldiversity.org>, which shows simulation results for different districts across the US.  The dashboard was built using the data dashboarding library Streamlit, and hence, includes only Python (and no typical browser/front-end) code.  

# Data

* [`data.zip`](https://plural-connections.s3.amazonaws.com/attendance-boundaries/data.zip) -> This should be unzipped and stored in the root folder of the repository.  It contains two main subfolders:
	- `derived_data/` -> this includes files per state output from the first four functions called in `utils/data_prep_pipeline.py` (described above).  Namely, it includes a file that represents an estimate of how many students per racial/ethnic group attend each each zoned (boundary-assigned) school from each school (`blocks_file_for_assignment.csv`).  This also constitutes the block-to-school mapping derived from the 21/22 school attendance boundaries data described in the paper.  **NOTE:** due to gaps in data availability in the common core of data, the `num_frl` (free/reduced lunch) and `num_ell` (English-language learner) estimates are unreliable, and hence, not used in the paper.  **These should be ignored.**  Due to licensing agreements, the folder does not include the raw shape files for attendance boundaries purchased from ATTOM.  It also does not include Census block shapefiles, which can be downloaded from the US Census Bureau's webpage.
	- `school_covariates/` -> this includes several files used for the school choice optout analysis (`school_choice_analysis.csv` -> produced by `analysis/analyze_districts.py`) and other district-level covariates used in the analyses described in the main text.  It also includes a file representing school latitudes and longitudes, sourced from the NCES.  Note: at the time of conducting analysis for the paper, we used a separate set of school latitudes and longitudes sourced from a data collaboration with GreatSchools.org for a prior project.  While most lat/longs should be the same, there may be slight discrepancies between these two datasets; if you find such discrepancies and/or they inhibit your ability to put this data to use for your purposes, please reach out.  
	- `prepped_csvs_for_analysis` -> this includes the resulting CSVs per district produced by running `analysis/analyze_model_solution.py` over a given set of simulation outputs found in `simulation_outputs/`.  These CSVs represent the expected impacts of each hypothetical rezoning (represented by a row in each CSV) on integration, travel times, and other outcomes of interest.  The CSV files are then loaded into `analysis/analyze_rezoning_sim_outputs.Rmd` for further processing and visualization.  This `.Rmd` file shows which columns are used, and how, to produce relevant plots and analyses for the paper.
* [`solver_files.zip`](https://plural-connections.s3.amazonaws.com/attendance-boundaries/solver_files.zip) -> This should be unzipped and stored in the `models/` folder.  The folder contains files needed to run `models/assignment_cp_sat.py` -- i.e., the rezoning algorithm.  In particular, there are three files per district, all of which are produced by `utils/model_utils.py`): 
	- `prepped_file_for_solver_<district_id>.csv`, which is a processed version of `blocks_file_for_assignment.csv`
	- `blocks_networks_<district_id>.json`, which represents a set of adjacency graphs, each rooted at a particular school in the district, that enables imposing contiguity constraints; and 
	- `prepped_travel_time_matrix_<district_id>.csv`, which represents a pre-computed travel times matrix (using the OpenRouteService API), indicating the estimated number of seconds it would take to drive from the centroid of each block in the district (rows) to each closed-enrollment elementary school (columns) without traffic.  
* [`simulation_outputs.zip`](https://plural-connections.s3.amazonaws.com/attendance-boundaries/simulation_outputs.zip) -> This should be unzipped and stored in the root folder of the repository.  It contains subfolders corresponding to simulation outputs for several different runs of the algorithm, which are described further below.  In each folder, simulation outputs are broken down by district.  Folder names within districts indicate the parameter configurations used to produce that particular output. For example, a folder named `0.75_0.15_white_min_total_segregation_0.0_False` represents a simulation with a max travel time increase of 75%; max school size increase of 15%; optimizing for White/non-White integration; 0% of existing neighboring blocks zoned to the same school as a given block being required to be rezoned to the same school as that block (see supplementary materials for more information on this parameter); and contiguity requirements set to False.  Within each folder like this, there is a `solution_*` file: each row in this file represents a block, and the column `new_school_nces` represents the school that a particular block is zoned for post simulation.  Below are the sets of results released with the paper: these folders can be referenced in `analysis/analyze_model_solution.py` to produce consolidated CSVs that are then, in turn, fed to the .Rmd script described above for further processing and plotting.
	- `2122_top_100_dissim_longer` -> rezoning results optimizing for White/non-White dissimilarity
	- `2122_top_100_gini` -> rezoning results optimizing for the White/non-White gini index
	- `2122_top_100_norm_exp` -> rezoning results optimizing for the White/non-White variance ratio (i.e., normalized exposure) index
	- `2122_top_100_norm_exp_sensitivity` -> additional simulations (optimizing for the normalized exposure index) representing sensitivity analyses (depicted in Figure 4 in the main text)
	- `2122_all_usa_dissim` -> rezoning results optimizing for White/non-White dissimilarity, expanded beyond the 98 in the paper to 4k+ districts across the US
