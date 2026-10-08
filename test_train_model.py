import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists("customer_churn_synthetic_raw.csv")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("customer_churn_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("customer_churn_model.pkl")

        sample = pd.DataFrame([{
            "Age": 35,
            "Gender": "Male",
            "TenureMonths": 24,
            "MonthlyCharges": 70.0,
            "ContractType": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "SupportCalls": 2,
            "LatePayments": 0,
            "PaperlessBilling": "Yes"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(int(prediction), [0, 1])

    def test_high_risk_customer(self):
        model = joblib.load("customer_churn_model.pkl")

        sample = pd.DataFrame([{
            "Age": 25,
            "Gender": "Male",
            "TenureMonths": 2,
            "MonthlyCharges": 95.0,
            "ContractType": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "SupportCalls": 8,
            "LatePayments": 5,
            "PaperlessBilling": "Yes"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(int(prediction), [0, 1])

    def test_low_risk_customer(self):
        model = joblib.load("customer_churn_model.pkl")

        sample = pd.DataFrame([{
            "Age": 45,
            "Gender": "Female",
            "TenureMonths": 60,
            "MonthlyCharges": 50.0,
            "ContractType": "Two year",
            "InternetService": "DSL",
            "PaymentMethod": "Bank transfer",
            "SupportCalls": 0,
            "LatePayments": 0,
            "PaperlessBilling": "No"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(int(prediction), [0, 1])


if __name__ == "__main__":
    unittest.main()
