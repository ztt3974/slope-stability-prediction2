import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import RobustScaler


class EnsemblePredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.scaler = None
        self.weights = None
        self.models = {}

    def load(self):
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.weights = joblib.load(os.path.join(self.model_dir, 'weights.pkl'))
        
        model_files = ['model_xgb.pkl', 'model_lgb.pkl', 'model_cat.pkl', 
                       'model_rf.pkl', 'model_et.pkl', 'model_gb.pkl']
        
        for mf in model_files:
            path = os.path.join(self.model_dir, mf)
            if os.path.exists(path):
                name = mf.replace('model_', '').replace('.pkl', '')
                self.models[name] = joblib.load(path)
        
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        
        weighted_proba = np.zeros(len(X), dtype=float)
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                proba = model.predict_proba(X_scaled)[:, 1]
                weighted_proba += proba * weight
        
        return weighted_proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)