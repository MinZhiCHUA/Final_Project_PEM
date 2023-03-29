import pandas as pd


def add_weight_for_data_balancing(
    df, label_col, weight_col, attribute_code_col, upper_qn, lower_qn
):

    df_count = (
        df.groupby([label_col, attribute_code_col])[attribute_code_col]
        .count()
        .reset_index(name="num_sample")
    )

    df_count[weight_col] = 1 / df_count["num_sample"]

    upper = df_count[weight_col].quantile(q=upper_qn)
    lower = df_count[weight_col].quantile(q=lower_qn)

    df_count[weight_col] = df_count[weight_col].clip(lower, upper)

    df = pd.merge(df, df_count, on=[attribute_code_col, label_col])

    return df
