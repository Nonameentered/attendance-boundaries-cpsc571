def add_leading_zero_for_dist(df, dist_id_col):
    return df[dist_id_col].str.rjust(7, "0")


def add_leading_zero_for_school(df, nces_id_col):
    return df[nces_id_col].str.rjust(12, "0")


def update_dist_id_with_leading_zero(dist_id):
    return dist_id.rjust(7, "0")
