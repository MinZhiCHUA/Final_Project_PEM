import pandas as pd
from sklearn.metrics import classification_report


def extract_att_lov(df, att_code_col_name, att_codes):

    df = df.loc[df[att_code_col_name].isin(att_codes)]
    return df


def remove_null_context(df, context_col_name):

    df = df.loc[df[context_col_name].notnull()]
    return df


def get_classification_report(label, label_pred):
    """
    ###############################################
    Function to obtain the classification report by comparing label and label_pred
    ###############################################
    input: labels in list
    output: dataframe with just the result and dataframe with details for each models
    """
    df = classification_report(label, label_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame.from_dict(df).transpose().reset_index()
    df = df.rename(columns={"support": "num_sample"})
    df_result = df.tail(3).reset_index(drop=True)
    df_result = df_result.rename(columns={"index": "metric"})
    df_data = (
        df.drop(df.tail(3).index)
        .sort_values(by=["num_sample"], ascending=False)
        .reset_index(drop=True)
    )
    df_data = df_data.rename(columns={"index": "model"})

    return df_result, df_data


def get_performance_metrics_table(list_df):
    """
    ###############################################
    Function to compile a dataframe from different dataframes with classification
    metrics (dataframe df_resutl produced by function get_classification_report)
    ###############################################
    input: distionary with title and dataframe, e.g., list = {'Overall': df_metric_overall, ...}
    output: dataframe with the compilation of all the  input dataframes
    """
    df_performance = pd.DataFrame()

    for key in list_df:
        df = list_df[key]
        data = {
            "Dataframe": [key],
            "Accuracy (%)": [
                round(df["precision"].loc[df["metric"] == "accuracy"].values[0] * 100, 2)
            ],
            "Precision (Weighted-avg)(%)": [
                round(
                    df["precision"].loc[df["metric"] == "weighted avg"].values[0] * 100,
                    2,
                )
            ],
            "Precision (Macro-avg) (%)": [
                round(df["precision"].loc[df["metric"] == "macro avg"].values[0] * 100, 2)
            ],
            "Recall (Weighted-avg) (%)": [
                round(df["recall"].loc[df["metric"] == "weighted avg"].values[0] * 100, 2)
            ],
            "Recall (Macro - avg) (%)": [
                round(df["recall"].loc[df["metric"] == "macro avg"].values[0] * 100, 2)
            ],
            "Number of samples": [
                round(df["num_sample"].loc[df["metric"] == "macro avg"].values[0])
            ],
        }
        df_new_row = pd.DataFrame(data=data)
        df_performance = pd.concat([df_performance, df_new_row])

    df_performance = df_performance.reset_index(drop=True)

    return df_performance
