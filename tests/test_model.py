"""Preprocessing, symptom vectors, model loading, prediction and ensemble maths.

Run from the project root:  python -m unittest discover -s tests -v
(requires trained artifacts: python -m training.train)
"""
import tempfile
import unittest

import numpy as np
import pandas as pd

import config
from backend.data_loader import load_raw_training_table, load_reference_data, match_key, normalize_disease, normalize_symptom
from backend.ensemble import soft_vote
from backend.predictor import DiseasePredictor, ModelArtifactError
from training import preprocess as pp

ARTIFACTS = all((config.MODELS_DIR / f"{k}.pkl").is_file() for k in config.MODEL_KEYS)


class TestPreprocessing(unittest.TestCase):
    def test_symptom_normalisation(self):
        self.assertEqual(normalize_symptom(" dischromic _patches"), "dischromic_patches")
        self.assertEqual(normalize_symptom("Skin Rash"), "skin_rash")
        self.assertEqual(normalize_symptom("foul_smell_of urine"), "foul_smell_of_urine")
        self.assertEqual(normalize_disease("Diabetes "), "Diabetes")
        self.assertEqual(normalize_disease("(vertigo) Paroymsal  Positional Vertigo"), "(vertigo) Paroymsal Positional Vertigo")

    def test_join_key_ignores_punctuation(self):
        self.assertEqual(match_key("foul_smell_ofurine"), match_key("foul_smell_of_urine"))

    def test_clean_table_removes_duplicates_and_padding(self):
        raw = pd.DataFrame({
            "Disease": ["Flu ", "Flu", "Flu", "Cold"],
            "Symptom_1": [" fever", "fever", "cough", "cough"],
            "Symptom_2": ["cough", "cough", np.nan, np.nan],
            "Symptom_3": [np.nan] * 4,
        })
        clean, stats = pp.clean_training_table(raw)
        self.assertEqual(len(clean), 3)  # first two rows are identical after cleaning
        self.assertEqual(stats["exact_duplicates_dropped"], 1)
        self.assertTrue(all(isinstance(t, tuple) for t in clean["symptoms"]))
        self.assertIn("Flu", set(clean["disease"]))

    def test_vocabulary_and_vectors(self):
        vocab = pp.build_vocabulary([("b", "a"), ("c",)])
        self.assertEqual(vocab, ["a", "b", "c"])
        vec = pp.symptoms_to_vector(["c", "a", "unknown"], vocab)
        self.assertEqual(vec.tolist(), [1, 0, 1])
        self.assertEqual(pp.vectorize([("a",), ("b", "c")], vocab).tolist(), [[1, 0, 0], [0, 1, 1]])

    def test_real_dataset_cleaning_and_split(self):
        clean, stats = pp.clean_training_table(load_raw_training_table())
        self.assertEqual(clean["disease"].nunique(), 41)
        self.assertEqual(stats["ambiguous_symptom_patterns"], 0)
        train, test = pp.split_train_test(clean)
        self.assertEqual(len(train) + len(test), len(clean))
        # no identical (disease, symptom-set) record may appear in both splits
        self.assertFalse(set(zip(train.disease, train.symptoms)) & set(zip(test.disease, test.symptoms)))
        # reproducible
        train2, _ = pp.split_train_test(clean)
        self.assertEqual(list(train.index), list(train2.index))

    def test_reference_join_covers_every_disease(self):
        clean, _ = pp.clean_training_table(load_raw_training_table())
        ref = load_reference_data(sorted(clean["disease"].unique()))
        for d in clean["disease"].unique():
            self.assertTrue(ref.description_for(d), f"no description for {d}")
            self.assertTrue(ref.precautions_for(d), f"no precautions for {d}")


class TestEnsembleMaths(unittest.TestCase):
    def test_equal_weight_is_mean_of_probabilities(self):
        a, b = np.array([[0.6, 0.4]]), np.array([[0.2, 0.8]])
        out = soft_vote({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
        np.testing.assert_allclose(out, [[0.4, 0.6]])
        self.assertEqual(out.argmax(), 1)  # soft vote can overturn a hard-vote tie

    def test_weighted_vote_sums_to_one(self):
        p = {k: np.random.default_rng(i).dirichlet(np.ones(5), size=1) for i, k in enumerate("abc")}
        out = soft_vote(p, {"a": 0.5, "b": 0.3, "c": 0.2})
        self.assertAlmostEqual(float(out.sum()), 1.0)

    def test_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            soft_vote({"a": np.ones((1, 2)) / 2}, {"a": 0.7})


@unittest.skipUnless(ARTIFACTS, "run `python -m training.train` first")
class TestTrainedArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = DiseasePredictor()

    def test_loading_and_consistency(self):
        self.assertEqual(len(self.pred.classes), 41)
        self.assertEqual(self.pred.vocabulary, sorted(self.pred.vocabulary))
        for key, model in self.pred.models.items():
            n_in = getattr(model, "n_features_in_", None) or model[-1].n_features_in_
            self.assertEqual(n_in, len(self.pred.vocabulary), key)
        self.assertAlmostEqual(sum(self.pred.weights.values()), 1.0)
        self.assertIn(self.pred.mode, ("equal", "performance_weighted"))

    def test_every_model_predicts_valid_distribution(self):
        probas = self.pred.model_probabilities(["itching", "skin_rash"])
        self.assertEqual(set(probas), set(config.MODEL_KEYS))
        for key, p in probas.items():
            self.assertEqual(p.shape, (1, 41), key)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=5, msg=key)

    def test_ensemble_equals_manual_weighted_average(self):
        symptoms = ["fatigue", "weight_loss", "polyuria"]
        probas = self.pred.model_probabilities(symptoms)
        manual = sum(self.pred.weights[k] * probas[k][0] for k in probas)
        result = self.pred.predict(symptoms)
        self.assertEqual(result["predicted_disease"], self.pred.classes[int(manual.argmax())])
        self.assertAlmostEqual(result["ensemble_confidence"], float(manual.max()), places=9)

    def test_prediction_payload(self):
        r = self.pred.predict(["itching", "skin_rash", "nodal_skin_eruptions"])
        self.assertEqual(r["predicted_disease"], "Fungal infection")
        self.assertEqual(len(r["model_predictions"]), 6)
        self.assertTrue(r["description"] and r["precautions"])
        self.assertIn("not a medical diagnosis", r["disclaimer"].lower())
        self.assertTrue(0 < r["ensemble_confidence"] <= 1)

    def test_artifacts_round_trip(self):
        again = DiseasePredictor()
        s = ["cough", "high_fever", "breathlessness"]
        self.assertEqual(self.pred.predict(s)["top_predictions"], again.predict(s)["top_predictions"])

    def test_missing_artifacts_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ModelArtifactError) as ctx:
                DiseasePredictor(tmp)
        self.assertNotIn(tmp, str(ctx.exception))  # no file-system paths leaked

    def test_unfitted_symptom_vector_is_order_stable(self):
        v1 = pp.symptoms_to_vector(["cough", "itching"], self.pred.vocabulary)
        v2 = pp.symptoms_to_vector(["itching", "cough"], self.pred.vocabulary)
        np.testing.assert_array_equal(v1, v2)


if __name__ == "__main__":
    unittest.main()
