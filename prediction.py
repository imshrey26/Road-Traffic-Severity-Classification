import joblib
import numpy as np
from sklearn.tree import ExtraTreeClassifier

def oridnal_encoder(input_val,feats):
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key,feat_val))
    value = feat_dict[input_val]
    return value

def get_prediction(data,model):
    """
    Predicts the class of a given datapoint
    """
    return model.predict(data)