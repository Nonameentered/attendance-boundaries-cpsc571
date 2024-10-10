from utils.header import *
from utils.aggregate_data import (
    output_block_to_school_mapping,
    add_in_demos_to_block_data_parallel,
)
from utils.output_school_and_block_info import (
    output_school_file_for_assignment,
    output_block_file_for_assignment_parallel,
)
from utils.model_utils import compute_solver_files_parallel

if __name__ == "__main__":
    output_block_to_school_mapping()
    add_in_demos_to_block_data_parallel()
    output_school_file_for_assignment()
    output_block_file_for_assignment_parallel()
    compute_solver_files_parallel()
