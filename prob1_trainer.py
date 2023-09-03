import pickle

import pandas as pd
import yaml
from lightgbm import LGBMClassifier

from data_processor import DataProcessor
from prob1 import config as cfg1

try:
    df = pd.read_parquet(cfg1.data_path)
except:
    df = pd.read_csv(cfg1.data_path)

processed_df = DataProcessor.process_data(df, cfg1)
target_column = "label"
# feature_columns = cfg1.feature_config["category_columns"] + cfg1.feature_config["numeric_columns"]
feature_columns = cfg1.feature_config["numeric_columns"]
X = processed_df[feature_columns]
print(X.shape)
y = processed_df[target_column]
prob1_config = "./prob1/model_config.yaml"
params_config = yaml.safe_load(open(prob1_config).read())
model = LGBMClassifier(**params_config["params_lgbm"])
model.fit(X, y, verbose=False)
pickle.dump(model, open("./prob1/model.pkl", "wb"))