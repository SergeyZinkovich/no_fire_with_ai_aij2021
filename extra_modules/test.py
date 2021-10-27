import warnings
import test_dataset_prepare
import solution
import helpers

warnings.simplefilter("ignore")


if __name__ == "__main__":
    val, val_targets = test_dataset_prepare.load_val_dataset()
    print("dataset prepared")
    result_df = solution.predict_output(val)
    print("output evaluated")
    result_df = result_df.drop("infire_day", axis=1)
    print(helpers.competition_metric(val_targets, result_df))
