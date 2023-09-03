import pickle
from category_encoders import CatBoostEncoder

class DataProcessor:
    @staticmethod
    def load_category_encoder(category_encoder_path: str):
        return pickle.load(open(category_encoder_path, "rb"))
    
    @staticmethod
    def save_category_encoder(category_encoder, category_encoder_path):
        pickle.dump(category_encoder, open(category_encoder_path, "wb"))

    @staticmethod
    def apply_category_features(df, category_columns=None, category_encoder=None):
        return df
        # df[category_columns] = category_encoder.transform(df[category_columns])
        # return df
    
    @staticmethod
    def process_data(df, cfg):
        category_columns = cfg.feature_config["category_columns"]
        Enc = CatBoostEncoder(cols=category_columns)
        try:
            Enc.fit_transform(df[category_columns], df[cfg.feature_config["target_column"]])
        except:
            Enc.fit_transform(df[category_columns], df[f'{cfg.feature_config["target_column"]}_encoded'])
        DataProcessor.save_category_encoder(Enc, cfg.category_index_path)
        encoded_df = df.copy()
        encoded_df[category_columns] = Enc.transform(df[category_columns])
        return encoded_df

    @staticmethod
    def apply_process_data(df, cfg, Enc):
        category_columns = cfg.feature_config["category_columns"]
        encoded_df = DataProcessor.apply_category_features(df, category_columns, Enc)
        return encoded_df