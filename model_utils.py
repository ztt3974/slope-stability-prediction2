import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


class OptimizedEnsemble:
    def __init__(self, input_size):
        self.input_size = input_size
        self.scaler = RobustScaler()
        self.models = {}
        self.weights = {}
        self.best_params = {}

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        
        probas = []
        weights = []
        
        for name, weight in self.weights.items():
            model = self.models[name]
            proba = model.predict_proba(X_scaled)[:, 1]
            probas.append(proba.astype(float))
            weights.append(weight)
        
        weighted_proba = np.zeros(len(X), dtype=float)
        for proba, weight in zip(probas, weights):
            weighted_proba += proba * weight
        
        return weighted_proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)