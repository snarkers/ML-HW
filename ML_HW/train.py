import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from preprocess import prep_data
from sklearn.model_selection import KFold

df = pd.read_csv(("fish_participant.csv"))

X, y = prep_data(df)

kf = KFold(n_splits = 20, shuffle = True, random_state = 42)
kf.get_n_splits(X)


for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    

wrf = RandomForestClassifier(class_weight = "balanced")
wrf.fit(X_train, y_train)
predict_y = wrf.predict(X_test)

dump(wrf, "wrf.joblib")