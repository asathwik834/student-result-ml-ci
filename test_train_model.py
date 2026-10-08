import os
import unittest
import joblib

from train_model import df, model, accuracy


class TestTrainModel(unittest.TestCase):

    def test_dataset_not_empty(self):
        self.assertGreater(len(df), 0)

    def test_required_columns_exist(self):
        required_columns = {
            "internal_marks",
            "attendance",
            "assignment_score",
            "result"
        }

        self.assertTrue(required_columns.issubset(df.columns))

    def test_result_is_binary(self):
        self.assertTrue(set(df["result"].unique()).issubset({0, 1}))

    def test_model_is_fitted(self):
        self.assertTrue(hasattr(model, "coef_"))

    def test_accuracy_is_valid(self):
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("model.joblib"))

    def test_model_can_be_loaded(self):
        loaded_model = joblib.load("model.joblib")
        self.assertTrue(hasattr(loaded_model, "predict"))


if __name__ == "__main__":
    unittest.main()
