# Solver constants
# MAX_SOLVER_TIME = 10200  # 2 hours 50 minutes (just under 3 to account for other stuff that might take time)
MAX_SOLVER_TIME = 19800  # 5 hours 30 minutes (just under 6 to account for other stuff that might take time e.g. loading data etc.)
# MAX_SOLVER_TIME = 39600  # 11 hours
NUM_SOLVER_THREADS = 1  # Only use the default SAT search

WHITE_KEY = {
    "num_white_to_school": "perwht",
}
BLACK_KEY = {
    "num_black_to_school": "perblk",
}

ASIAN_KEY = {
    "num_asian_to_school": "perasn",
}

NATIVE_KEY = {
    "num_native_to_school": "pernam",
}

HISP_KEY = {
    "num_hispanic_to_school": "perhsp",
}

ELL_KEY = {
    "num_ell_to_school": "perell",
}

FRL_KEY = {
    "num_frl_to_school": "perfrl",
}

TOTAL_KEY = {
    "num_total_to_school": "pertotal",
}

ALL_CATS = {
    "black": BLACK_KEY,
    "white": WHITE_KEY,
    "asian": ASIAN_KEY,
    "native": NATIVE_KEY,
    "hisp": HISP_KEY,
    "ell": ELL_KEY,
    "frl": FRL_KEY,
    "total": TOTAL_KEY,
}

# To scale variables so they satisfy cp-sat's integer requirements
SCALING = (100000, 5)

MAX_STUDENTS_PER_CAT = 2000
MAX_TOTAL_STUDENTS = 2000
