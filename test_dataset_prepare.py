import features_generation
import preprocessing
import os
import pandas as pd
from constants import *


def prepare_dataset(test):
    cities_df = preprocessing.prepare_cities(
        PATH_TO_INPUT, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, STEP
    )
    grib_list = [
        el.split(".")[0]
        for el in os.listdir(os.path.join(PATH_TO_INPUT, "ERA5_data"))
        if el.startswith(
            (
                "temp",
                "wind",
                "evaporation1",
                "evaporation2",
                "heat1",
                "heat2",
                "vegetation",
            )
        )
        and el.endswith("2021.grib")
    ]
    for file_name in grib_list:
        preprocessing.make_pool_features(
            os.path.join(PATH_TO_INPUT, "ERA5_data"), file_name, PATH_TO_ADD_DATA
        )
    test = features_generation.add_pooling_features(test, PATH_TO_ADD_DATA, count_lag=3)
    test = features_generation.add_cat_date_features(test)
    test = features_generation.add_geo_features(test, cities_df)

    return test


def load_val_dataset():
    val = pd.read_parquet(os.path.join(PATH_TO_INPUT, 'prepared_val.parquet'))
    targets = val.iloc[:, 11:11 + 8]
    return val, targets


def load_test_dataset():
    return prepare_dataset(pd.read_csv(os.path.join(PATH_TO_INPUT, "test.csv"), index_col="id", parse_dates=["dt"]))
