import os
import pickle
import warnings
import pandas as pd
import test_dataset_prepare
from constants import *

warnings.simplefilter("ignore")


def predict_output(test):
    result_df = pd.DataFrame()
    for idx in range(1, 9):
        path_to_model = os.path.join(PATH_TO_MODELS, f"model_{idx}_day.pkl")
        with open(path_to_model, "rb") as f:
            model = pickle.load(f)
        result_df[f"infire_day_{idx}"] = (
            model.predict_proba(test[FEATURES])[:, 1] > 0.59
        ).astype(int)

    with open(os.path.join(PATH_TO_MODELS, "model_mc.pkl"), "rb") as f:
        meta_model = pickle.load(f)
    result_df["infire_day"] = meta_model.predict(test[FEATURES])
    index_to_replace = result_df[
        result_df[[f"infire_day_{day}" for day in range(1, 9)]].sum(axis=1) == 0
    ].index
    for i in range(1, 9):
        result_df.loc[
            (result_df.index.isin(index_to_replace) & (result_df["infire_day"] == i)),
            f"infire_day_{i}",
        ] = 1

    return result_df


def save_output_to_csv(result_df):
    result_df.drop("infire_day", axis=1).to_csv(
        os.path.join(PATH_TO_OUTPUT, "output.csv"), index_label="id"
    )


if __name__ == "__main__":
    test = test_dataset_prepare.load_test_dataset()
    result_df = predict_output(test)
    save_output_to_csv(result_df)
