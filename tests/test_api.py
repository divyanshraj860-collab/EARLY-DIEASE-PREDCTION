"""REST API tests (Flask test client).  Requires trained artifacts."""
import unittest

import config
from app import create_app

ARTIFACTS = all((config.MODELS_DIR / f"{k}.pkl").is_file() for k in config.MODEL_KEYS)
VALID = {"age": 30, "gender": "Male", "symptoms": ["high_fever", "headache", "cough"]}


@unittest.skipUnless(ARTIFACTS, "run `python -m training.train` first")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def post(self, body, **kw):
        return self.client.post("/api/predict", json=body, **kw)

    def test_index_and_headers(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("default-src 'self'", r.headers["Content-Security-Policy"])
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")

    def test_health_and_symptoms(self):
        self.assertEqual(self.client.get("/api/health").json["status"], "ok")
        d = self.client.get("/api/symptoms").json
        self.assertEqual(d["count"], len(d["symptoms"]))
        self.assertTrue({"name", "label", "severity"} <= set(d["symptoms"][0]))

    def test_predict_success(self):
        body = {**VALID, "additional_info": {"height": 175, "weight": 70, "blood_group": "O+"}}
        r = self.post(body)
        self.assertEqual(r.status_code, 200)
        j = r.json
        for key in ("predicted_disease", "ensemble_confidence", "model_predictions", "description",
                    "precautions", "disclaimer", "top_predictions", "patient_info"):
            self.assertIn(key, j)
        self.assertEqual(set(j["model_predictions"]), set(config.MODEL_KEYS))
        self.assertEqual(j["patient_info"]["bmi"], 22.9)
        self.assertEqual(j["patient_info"]["age"], 30)

    def test_symptom_names_are_normalised(self):
        r = self.post({**VALID, "symptoms": ["Skin Rash", "ITCHING", "itching"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual([s["name"] for s in r.json["selected_symptoms"]], ["skin_rash", "itching"])

    def test_invalid_inputs(self):
        cases = [
            ({**VALID, "symptoms": []}, "symptoms"),
            ({**VALID, "symptoms": ["not_a_symptom"]}, "symptoms"),
            ({**VALID, "symptoms": "fever"}, "symptoms"),
            ({**VALID, "age": -5}, "age"),
            ({**VALID, "age": 500}, "age"),
            ({**VALID, "age": "abc"}, "age"),
            ({**VALID, "age": True}, "age"),
            ({"gender": "Male", "symptoms": ["cough"]}, "age"),
            ({**VALID, "gender": "robot"}, "gender"),
            ({**VALID, "additional_info": {"height": 5000}}, "height"),
            ({**VALID, "additional_info": {"blood_group": "Z+"}}, "blood_group"),
            ({**VALID, "additional_info": {"allergies": "x" * 1000}}, "allergies"),
        ]
        for body, field in cases:
            r = self.post(body)
            self.assertEqual(r.status_code, 400, body)
            self.assertEqual(r.json.get("field"), field, body)
            self.assertIn("error", r.json)

    def test_unknown_symptom_gets_suggestions(self):
        r = self.post({**VALID, "symptoms": ["fever"]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("high_fever", r.json["error"])

    def test_malformed_requests(self):
        self.assertEqual(self.client.post("/api/predict", data="{bad", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="x", content_type="text/plain").status_code, 415)
        self.assertEqual(self.post([1, 2, 3]).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="", content_type="application/json").status_code, 400)
        big = self.client.post("/api/predict", data="x" * 100_000, content_type="application/json")
        self.assertEqual(big.status_code, 413)
        self.assertEqual(self.client.get("/api/predict").status_code, 405)

    def test_text_is_sanitised(self):
        r = self.post({**VALID, "additional_info": {"allergies": "<script>alert(1)</script>"}})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("<", r.json["patient_info"]["allergies"])

    def test_disease_endpoint(self):
        r = self.client.get("/api/disease/Malaria")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json["description"] and r.json["precautions"])
        self.assertEqual(self.client.get("/api/disease/Nonexistent").status_code, 404)

    def test_model_performance_and_images(self):
        r = self.client.get("/api/model-performance")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json["models"]), set(config.MODEL_KEYS))
        self.assertEqual(self.client.get("/api/report-image/confusion_matrix_svc").status_code, 200)
        self.assertEqual(self.client.get("/api/report-image/..%2Fconfig").status_code, 404)
        self.assertEqual(self.client.get("/api/report-image/secret").status_code, 404)

    def test_unknown_route_is_json(self):
        r = self.client.get("/api/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.json)


class TestMissingModels(unittest.TestCase):
    def test_api_degrades_gracefully(self):
        client = create_app(load_models=False).test_client()
        self.assertEqual(client.get("/api/health").status_code, 503)
        r = client.post("/api/predict", json=VALID)
        self.assertEqual(r.status_code, 503)
        self.assertIn("train", r.json["error"])
        self.assertEqual(client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
