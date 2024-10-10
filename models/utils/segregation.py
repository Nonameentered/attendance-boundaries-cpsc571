from numpy.core.shape_base import block
from utils.header import *


def compute_seg(qsj, Qj):

    # Special casing this to avoid a divide by zero error
    if Qj.iloc[0] == 1:
        Qj = 1 - np.finfo(float).eps
    seg = np.nan_to_num((float(qsj) - float(Qj)) / (1 - float(Qj)))
    return seg


def compute_school_segregation(school, district):

    groups = [
        "perblk",
        "pernam",
        "perasn",
        "perhsp",
        "perfrl",
        "perell",
        "perwht",
    ]
    seg_indices = {}
    for g in groups:
        seg_indices[g] = compute_seg(school[g], district[g])
    return seg_indices


def get_school_total_for_cat(curr_school, cat_key_for_r):
    val = "total_" + cat_key_for_r.split("_to_school")[0].split("num_")[1]
    key = val
    if val == "total_hisp":
        key = "total_hispanic"
    if val == "total_total":
        key = "total_enrollment"

    # In case there are negatives, make zero the min
    # Also, fill in nans as zeros.  Interpolation might make sense,
    # But there isn't a clear winner in terms of interpolation strategy
    # So setting to zero for now
    cat_total_at_school = int(max(np.nan_to_num(pd.to_numeric(curr_school[key])), 0))
    total_at_school = int(
        max(np.nan_to_num(pd.to_numeric(curr_school["total_enrollment"])), 0)
    )
    return min(cat_total_at_school, total_at_school)


def block_allocation(curr_school, curr_blocks, r, block_students_by_cat, cat_keys):
    total_students_to_allocate = get_school_total_for_cat(curr_school, cat_keys[r])

    key = cat_keys[r].split("_to_school")[0]
    perc_key = "curr_percent_{}".format(key)
    curr_blocks[perc_key] = (
        pd.to_numeric(curr_blocks[key]) / pd.to_numeric(curr_blocks[key]).sum()
    )
    curr_blocks = curr_blocks.sort_values(by=perc_key, ascending=False).reset_index()
    num_students_remaining = total_students_to_allocate

    # If we have a high fraction of the given category of
    # students at the school, it's possible our
    # census-based block-level counts are off.  So
    # instead, simply assume the number of students in that
    # category hailing from each block to the school
    # is proportional to the total number of students
    # we estimate to be living in that block
    perc_total_key = "percent_total_pop_in_block"
    curr_blocks[perc_total_key] = (
        pd.to_numeric(curr_blocks["num_total"])
        / pd.to_numeric(curr_blocks["num_total"]).sum()
    )

    perc_key_to_use = perc_key
    if pd.to_numeric(curr_school[r]) > 0.5:
        perc_key_to_use = perc_total_key

    num_allocated = 0
    for i in range(0, len(curr_blocks)):
        if num_students_remaining <= 0:
            num_to_allocate = 0
        else:
            num_to_allocate = int(
                min(
                    num_students_remaining,
                    np.ceil(
                        np.nan_to_num(
                            total_students_to_allocate * curr_blocks[perc_key_to_use][i]
                        )
                    ),
                )
            )
        block_students_by_cat[curr_blocks["block_id"][i]][
            cat_keys[r]
        ] += num_to_allocate
        num_allocated += num_to_allocate
        num_students_remaining -= num_to_allocate

    # print(r, total_students_to_allocate, num_allocated)
    return num_allocated


def allocate_students_to_blocks(curr_school, curr_blocks, all_cat_keys):

    block_students_by_cat = defaultdict(Counter)
    for cats in all_cat_keys:
        for k in cats:
            block_allocation(
                curr_school,
                curr_blocks,
                k,
                block_students_by_cat,
                cats,
            )

    return block_students_by_cat


if __name__ == "__main__":
    pass
