import joblib
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--value", required=True,
	help="value of prediction")
args = vars(ap.parse_args())

scaler = joblib.load('MaxAbsScaler.pkl')

pred = args["value"]
y_pred = float(pred)
y_pred = np.full((1, 1), y_pred)
y_pred = y_pred.reshape(-1, 1)

y_pred = scaler.inverse_transform(y_pred)
y_pred = y_pred.flatten()

print(y_pred)