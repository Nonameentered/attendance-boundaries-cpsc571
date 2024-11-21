# Be sure to sync the S3 data locally here, per ../README.md
DATA_DIR = "../simulation_outputs/"
# DATA_DIR = "../"

TITLE = "Increasing school diversity"
ROOT_URL = "https://www.schooldiversity.org"
OPTIMIZATION_TARGETS = {
    "White students": "white",
    # "free and reduced lunch students": "frl",
}
OUTCOME_TARGETS = {
    "White students": "white",
    "Black students": "black",
    "Hispanic students": "hisp",
    "Asian students": "asian",
    "Native American students": "native",
    # "free and reduced lunch students": "frl",
    # "English language learners": "ell",
}
OPTIMIZATION_TARGETS_DISPLAY = {x: y for y, x in OPTIMIZATION_TARGETS.items()}
OUTCOME_TARGETS_DISPLAY = {x: y for y, x in OUTCOME_TARGETS.items()}

DEFAULT_TARGET_GROUP = list(OPTIMIZATION_TARGETS.values())[0]
QUALTRICS_SURVEY_URL = "https://mit.co1.qualtrics.com/jfe/form/SV_1H3WTHJbJLv6v8a?session_id={session_id}&source={source_id}&travel_time_threshold={travel_time_threshold}&school_size_threshold={school_size_threshold}&student_cat={student_cat}&obj_fcn={obj_fcn}&is_contiguous={is_contiguous}&referral_code={referral_code}&set={set}"

# Config code indicating the status quo
STATUS_QUO_ZONING = "status_quo_zoning"

# First color is used to represent "status quo", second is "proposed", third is "ideal"
COLORSCALE = ["pink", "lightgreen", "lightblue"]  # may also be specified as rgb(1,2,3)

# The suffix of the column for which we pick the target-group-specific argmax configuration to display by default
OBJECTIVE_COLUMN = "_total_segregation_diff"

# Subdirectory in which to write maps cache files (relative to frontend dir)
MAPS_CACHE_DIR = "cache"

# Category mapping
CAT_MAPPING = {
    "frl": "perfrl",
    "black": "perblk",
    "white": "perwht",
    "hispanic": "perhsp",
    "native": "pernam",
    "asian": "perasn",
    "ell": "perell",
}

EMAIL = "schooldiversity@media.mit.edu"


def config_code_qualifies_for_default(code):
    """Determines whether a given configuration qualifies as a default configuration."""
    parts = parse_config_code(code)
    return (
        code != STATUS_QUO_ZONING
        and parts["travel"] <= 0.5
        and parts["size"] <= 0.15
        and parts["cohesion"] == 0.5
        and parts["is_contiguous"] == "True"
    )


def get_permalink(state, district_id, target_group, config_code, default_config_code):
    url = ROOT_URL + ("/?state=%s&district=%s" % (state, district_id))
    if config_code != default_config_code:
        url += "&config_code=" + config_code
    if target_group != DEFAULT_TARGET_GROUP:
        url += "&target_group=" + target_group
    return url


def get_referral_code_link(referral_code):
    return ROOT_URL + "?referral_code=%s" % (referral_code)


def parse_config_code(code):
    if code == STATUS_QUO_ZONING:
        return {}
    parts = code.split("_")
    return {
        "travel": float(parts[0]),
        "size": float(parts[1]),
        "cohesion": float(parts[-2]),
        "is_contiguous": parts[-1],
    }
