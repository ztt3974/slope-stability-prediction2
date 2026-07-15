import numpy as np
import pandas as pd
import joblib
import os

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               GradientBoostingClassifier)


class EnsemblePredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.scaler = None
        self.weights = None
        self.models = {}

    def load(self):
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.weights = joblib.load(os.path.join(self.model_dir, 'weights.pkl'))
        
        xgb_path = os.path.join(self.model_dir, 'model_xgb.json')
        if os.path.exists(xgb_path):
            self.models['xgb'] = xgb.XGBClassifier()
            self.models['xgb'].load_model(xgb_path)
        
        cat_path = os.path.join(self.model_dir, 'model_cat.cbm')
        if os.path.exists(cat_path):
            self.models['cat'] = CatBoostClassifier()
            self.models['cat'].load_model(cat_path)
        
        lgb_path = os.path.join(self.model_dir, 'model_lgb.pkl')
        if os.path.exists(lgb_path):
            self.models['lgb'] = joblib.load(lgb_path)
        
        rf_path = os.path.join(self.model_dir, 'model_rf.pkl')
        if os.path.exists(rf_path):
            self.models['rf'] = joblib.load(rf_path)
        
        et_path = os.path.join(self.model_dir, 'model_et.pkl')
        if os.path.exists(et_path):
            self.models['et'] = joblib.load(et_path)
        
        gb_path = os.path.join(self.model_dir, 'model_gb.pkl')
        if os.path.exists(gb_path):
            self.models['gb'] = joblib.load(gb_path)
        
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