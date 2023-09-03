import json
import os
import warnings
from abc import ABC, abstractmethod
import pickle
import daal4py as d4p
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from data_processor import DataProcessor
from prob1 import config as cfg1
from prob2 import config as cfg2
from fastapi_profiler import PyInstrumentProfilerMiddleware
from typing import Dict
import orjson
from data_drift import detect_drift_prob1, detect_drift_prob2
import pyinstrument

warnings.filterwarnings("ignore")
PORT = 5040


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class BaseModelPredictor(ABC):
    @abstractmethod
    def process_data(self, data: Data):
        pass

    @abstractmethod
    def predict(self, data: Data):
        pass


class ModelPredictorProb1(BaseModelPredictor):
    def __init__(self):
        self.categorical_cols = cfg1.feature_config["category_columns"]
        self.category_encoder = DataProcessor.load_category_encoder(cfg1.category_index_path)
        # self.feature_columns = cfg1.feature_config["category_columns"] + cfg1.feature_config["numeric_columns"]
        self.feature_columns = cfg1.feature_config["numeric_columns"]
        self.model = pickle.load(open(cfg1.model_path, "rb"))
        self.daal_model = d4p.get_gbt_model_from_lightgbm(self.model.booster_)
        self.predictions_container = d4p.gbt_classification_prediction(
            nClasses=2, resultsToEvaluate='computeClassProbabilities', fptype='float')
        self.previous_shape = 0
        try:
            self.ref_df = pd.read_parquet(cfg1.original_data_path)
        except:
            self.ref_df = pd.read_csv(cfg1.original_data_path)
        self.ref_df = DataProcessor.apply_process_data(self.ref_df, cfg1, self.category_encoder)
        self.ref_df = self.ref_df.loc[:, self.feature_columns]

    def process_data(self, data: Data):
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = DataProcessor.apply_process_data(raw_df, cfg1, self.category_encoder)
        features = feature_df[self.feature_columns]
        return features

    async def daal_predict(self, data: Data):
        features = self.process_data(data)
        if self.previous_shape != len(data.rows):
            self.predictions_container = d4p.gbt_classification_prediction(
                nClasses=2, resultsToEvaluate='computeClassProbabilities', fptype='float')
            self.previous_shape = len(data.rows)
        preds = self.predictions_container.compute(features, self.daal_model).probabilities[:, 1]
        return {
            "id": data.id,
            "predictions": preds.tolist(),
            "drift": int(detect_drift_prob1(self.ref_df, features))
        }

    def predict(self, data: Data):
        features = self.process_data(data)
        predictions = self.model.predict_proba(features)[:, 1]
        return {
            "id": data.id,
            "predictions": predictions.tolist(),
            "drift": 0
        }


class ModelPredictorProb2(BaseModelPredictor):
    def __init__(self):
        self.categorical_cols = cfg2.feature_config["category_columns"]
        self.category_encoder = DataProcessor.load_category_encoder(cfg2.category_index_path)
        # self.feature_columns = cfg2.feature_config["category_columns"] + cfg2.feature_config["numeric_columns"]
        self.feature_columns = cfg2.feature_config["numeric_columns"]
        self.model = pickle.load(open(cfg2.model_path, "rb"))
        self.classes = self.model.classes_
        self.daal_model = d4p.get_gbt_model_from_lightgbm(self.model.booster_)
        self.predictions_container = d4p.gbt_classification_prediction(
            nClasses=6, resultsToEvaluate='computeClassLabels', fptype='float')
        self.previous_shape = 0
        try:
            self.ref_df = pd.read_parquet(cfg2.original_data_path)
        except:
            self.ref_df = pd.read_csv(cfg2.original_data_path)
        self.ref_df = DataProcessor.apply_process_data(self.ref_df, cfg2, self.category_encoder)
        self.ref_df = self.ref_df.loc[:, self.feature_columns]

    def process_data(self, data: Data):
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = DataProcessor.apply_process_data(raw_df, cfg2, self.category_encoder)
        features = feature_df.loc[:, self.feature_columns]
        return features

    async def daal_predict(self, data: Data):
        features = self.process_data(data)
        if self.previous_shape != len(data.rows):
            self.predictions_container = d4p.gbt_classification_prediction(
                nClasses=6, resultsToEvaluate='computeClassLabels', fptype='float')
            self.previous_shape = len(data.rows)
        preds = self.predictions_container.compute(features, self.daal_model).prediction[:, 0]
        return {
            "id": data.id,
            "predictions": [self.classes[i] for i in list(map(int, preds.tolist()))],
            "drift": int(detect_drift_prob2(self.ref_df, features))
        }

    def predict(self, data: Data):
        features = self.process_data(data)
        predictions = self.model.predict(features)
        return {
            "id": data.id,
            "predictions": predictions.tolist(),
            "drift": 0
        }


class PredictorApi:
    def __init__(self, predictor_prob1: ModelPredictorProb1, predictor_prob2: ModelPredictorProb2):
        self.predictor_prob1 = predictor_prob1
        self.predictor_prob2 = predictor_prob2
        self.app = FastAPI()
        # self.app.add_middleware(
        #                         PyInstrumentProfilerMiddleware,
        #                         server_app=self.app,  # Required to output the profile on server shutdown
        #                         profiler_output_type="html",
        #                         is_print_each_request=True,  # Set to True to show request profile on
        #                                                     # stdout on each request
        #                         open_in_browser=False,  # Set to true to open your web-browser automatically
        #                                                 # when the server shuts down
        #                         html_file_name="example_profile.html"  # Filename for output
        #                     )

        @self.app.get("/")
        async def root():
            return {"health": "ok"}

        @self.app.post("/phase-3/prob-1/predict")
        async def predict_prob1(request: Request):
            # profiler = pyinstrument.Profiler()
            # profiler.start()
            body = b"".join([data async for data in request.stream()])
            data = Data(**orjson.loads(body))
            response = await self.predictor_prob1.daal_predict(data)
            # self._log_request(data, "prob1")
            # self._log_response(response, "prob1")
            # profiler.stop()
            # with open(f"./profiling/prob1/{data.id}.html", 'w') as f:
            #     f.write(profiler.output_html())
            return ORJSONResponse(response)

        @self.app.post("/phase-3/prob-2/predict")
        async def predict_prob2(request: Request):
            # profiler = pyinstrument.Profiler()
            # profiler.start()
            body = b"".join([data async for data in request.stream()])
            data = Data(**orjson.loads(body))
            response = await self.predictor_prob2.daal_predict(data)
            # self._log_request(data, "prob2")
            # self._log_response(response, "prob2")
            # profiler.stop()
            # with open(f"./profiling/prob1/{data.id}.html", 'w') as f:
            #     f.write(profiler.output_html())
            return ORJSONResponse(response)

    @staticmethod
    def _log_request(data: Data, problem_id: str):
        my_dict = {
            "id": data.id,
            "rows": data.rows,
            "columns": data.columns
        }
        with open(f"./save_request_data/{problem_id}/{data.id}.json", "w") as file:
            json.dump(my_dict, file)

    @staticmethod
    def _log_response(response, problem_id: str):
        with open(f"./results/{problem_id}/{response['id']}.json", "w") as file:
            json.dump(response, file)

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


os.makedirs("./save_request_data/prob1", exist_ok=True)
os.makedirs("./save_request_data/prob2", exist_ok=True)
predictor_prob1 = ModelPredictorProb1()
predictor_prob2 = ModelPredictorProb2()
api = PredictorApi(predictor_prob1=predictor_prob1,
                   predictor_prob2=predictor_prob2)

server = api.app
