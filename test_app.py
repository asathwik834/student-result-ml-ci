import unittest

from app import app


class CustomerChurnAPITest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertEqual(data["status"], "ok")
        self.assertEqual(
            data["service"],
            "customer-churn-prediction"
        )

    def test_prediction_endpoint(self):
        customer = {
            "Age": 35,
            "Gender": "Male",
            "TenureMonths": 12,
            "MonthlyCharges": 75.50,
            "ContractType": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "SupportCalls": 3,
            "LatePayments": 1,
            "PaperlessBilling": "Yes"
        }

        response = self.client.post(
            "/predict",
            json=customer
        )

        # INTENTIONALLY CHANGED TO 500
        # This is for the controlled failure demonstration.
        self.assertEqual(response.status_code, 500)

        data = response.get_json()

        self.assertIn("prediction", data)
        self.assertIn("prediction_code", data)

        self.assertIn(
            data["prediction"],
            ["CHURN", "NO CHURN"]
        )

        self.assertIn(
            data["prediction_code"],
            [0, 1]
        )

    def test_missing_fields(self):
        customer = {
            "Age": 35,
            "Gender": "Male"
        }

        response = self.client.post(
            "/predict",
            json=customer
        )

        self.assertEqual(response.status_code, 400)

        data = response.get_json()

        self.assertIn("error", data)
        self.assertIn("missing_fields", data)


if __name__ == "__main__":
    unittest.main()
