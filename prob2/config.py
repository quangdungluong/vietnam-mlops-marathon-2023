data_path = "./prob2/data/cleaned_combined_data.csv"
original_data_path = "./phase-3/prob-2/raw_train.parquet"
model_path = "./prob2/model.pkl"
category_index_path = "./prob2/category_encoder.pkl"
feature_config = {
    "numeric_columns": [
        "feature1",
        "feature5",
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
        "feature14",
        "feature15",
        "feature16",
        "feature17",
        "feature18",
        "feature19",
        "feature20",
        "feature21",
        "feature22",
        "feature23",
        "feature24",
        "feature25",
        "feature26",
        "feature27",
        "feature28",
        "feature29",
        "feature30",
        "feature31",
        "feature32",
        "feature33",
        "feature34",
        "feature35",
        "feature36",
        "feature37",
        "feature38",
        "feature39",
        "feature40",
        "feature41"
    ],
    "category_columns": [
        "feature2",
        "feature3",
        "feature4"
    ],
    "target_column": "label",
    "ml_type": "classification"
}