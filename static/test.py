import pandas as pd
import numpy as np
import sklearn.impute as impute
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as pl
import joblib

dlrfc = joblib.load("../tempfiles/test.joblib")

# dlrfc.train("../data_new/train_10000.csv")

dlrfc.validate("../data_new/validate_1000.csv")

dlrfc.test("../data_new/test_2000_x.csv")