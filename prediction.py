import joblib
import numpy as np


def predict(data):
    clf = joblib.load("best.sav.gz")

    return clf.predict(data)