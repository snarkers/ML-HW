
### YOU WRITE THIS ###
import os
from joblib import load
from preprocess import prep_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preprocess import prep_data
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    X, y = prep_data(df)

    wrf = load("wrf.joblib")

    predictions = wrf.predict(X)

    return predictions

if __name__ == "__main__":
    predictions = predict_from_csv(("fish_holdout_demo.csv"))
    print(predictions)
    print(pd.read_csv("fish_holdout_demo.csv")["Weight"].values)
######

######

# ### WE WRITE THIS ###
#     from sklearn.metrics import mean_squared_error
#     ho_predictions = predict_from_csv("fish_holdout.csv")
#     ho_truth = pd.read_csv("fish_holdout.csv")["Weight"].values
#     ho_mse = mean_squared_error(ho_truth, ho_predictions)
#     print(ho_mse)
# ######
© 2020 GitHub, Inc.