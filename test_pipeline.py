# test_pipeline.py
import joblib
import numpy as np

def test_model_loading():
    modelo = joblib.load("models/melhor_modelo_multimodal.pkl")
    assert modelo is not None

def test_prediction_shape():
    modelo = joblib.load("models/melhor_modelo_multimodal.pkl")
    exemplo = np.random.rand(1, modelo.n_features_in_)
    prob = modelo.predict_proba(exemplo)[0, 1]
    assert 0 <= prob <= 1
