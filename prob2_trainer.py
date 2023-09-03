import pickle

import pandas as pd
import yaml
from lightgbm import LGBMClassifier

from data_processor import DataProcessor
from prob2 import config as cfg2
from sklearn.preprocessing import LabelEncoder

try:
    df = pd.read_parquet(cfg2.data_path)
except:
    df = pd.read_csv(cfg2.data_path)
    
le = LabelEncoder()
le.fit(df['label'])
df['label_encoded'] = le.transform(df['label'])
processed_df = DataProcessor.process_data(df, cfg2)
target_column = "label"
# feature_columns = cfg2.feature_config["category_columns"] + cfg2.feature_config["numeric_columns"]
feature_columns = cfg2.feature_config["numeric_columns"]
X = processed_df[feature_columns]
y = processed_df[target_column]
prob1_config = "./prob2/model_config.yaml"
params_config = yaml.safe_load(open(prob1_config).read())
model = LGBMClassifier(**params_config["params_lgbm"])
model.fit(X, y, verbose=False)
pickle.dump(model, open("./prob2/model.pkl", "wb"))