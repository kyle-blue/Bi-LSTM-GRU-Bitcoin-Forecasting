import os
from app.DataPreprocesser import DataPreprocesser
from app.Model import Model
from app.get_indicators import get_select_indicator_values
from app.optimise_params import create_tf_session, optimise_params
from app.parameters import Architecture, Symbol
from app.test_model import test_model
from .indicator_correlations import indicator_correlations
import tensorflow.keras.backend as K


def multi_test(SYMBOL_TO_PREDICT: str):
    params = {
        "architecture": Architecture.LSTM.value,
        "is_bidirectional": True,
        "indicators": False,
        "sequence_length": 200,
        "forecast_period": 1,
    }

    tests = [
        (5, {"architecture": Architecture.LSTM.value, "is_bidirectional": False,}),
        (5, {"architecture": Architecture.GRU.value, "is_bidirectional": False,}),
        (5, {"architecture": Architecture.LSTM.value, "is_bidirectional": True,}),
        (5, {"architecture": Architecture.GRU.value, "is_bidirectional": True,}),
        (5, {"indicators": False}),
        (5, {"indicators": True}),
        (1, {"sequence_length": 50}),
        (1, {"sequence_length": 100}),
        (1, {"sequence_length": 150}),
        (1, {"sequence_length": 200}),
        (1, {"sequence_length": 250}),
        (1, {"sequence_length": 300}),
        (1, {"sequence_length": 350}),
        (1, {"sequence_length": 400}),
        (1, {"forecast_period": 1}),
        (1, {"forecast_period": 5}),
        (1, {"forecast_period": 10}),
        (1, {"forecast_period": 20}),
        (1, {"forecast_period": 30}),
    ]

    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto/{SYMBOL_TO_PREDICT}.parquet",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        sequence_length=200,
        should_ask_load=False
    )
    preprocessor.preprocess()

    folder = f"{os.environ['WORKSPACE']}/results/tests"
    for test_num, test in enumerate(tests):
        test_num = test_num
        num_repeats = test[0]
        additional_params = test[1]
        new_params = {**params, **additional_params}

        pre = None
        try: pre = preprocessor
        except: pass
        if "sequence_length" in additional_params or "forecast_period" in additional_params or "indicators" in additional_params:
            try:
                del preprocessor
                del pre
            except: pass
            pre = DataPreprocesser(
                f"{os.environ['WORKSPACE']}/data/crypto/{SYMBOL_TO_PREDICT}.parquet",
                col_names=["open", "high", "low", "close", "volume"],
                forecast_col_name="close",
                forecast_period=new_params["forecast_period"],
                sequence_length=new_params["sequence_length"],
                should_ask_load=False
            )
            pre.preprocess()
            if new_params["indicators"]:
                indicator_df = get_select_indicator_values(pre.df_original)
                pre.change_data(indicator_df)
                pre.print_df()

        for i in range(num_repeats):
            create_tf_session()
            print(f"Test {test_num} repeat {i}")
            print("Params")
            print(new_params)

            train_x, train_y = pre.get_train()
            validation_x, validation_y = pre.get_validation()
            model = Model(
                train_x, train_y, validation_x, validation_y,
                pre.get_seq_info_str(),
                architecture=new_params["architecture"],
                is_bidirectional=new_params["is_bidirectional"],
                batch_size=1024,
                hidden_layers=2,
                neurons_per_layer=100,
                dropout=0.2,
                initial_learn_rate=0.001
            )
            model.train()
            r_square = model.score["RSquaredMetric"]
            mae = model.score["mae"]
            train_time = float(model.train_time) / 60.0 # In Minutes
            
            file_path = f"{folder}/test{test_num}.csv"

            if not os.path.exists(file_path):
                with open(file_path, 'a') as file:
                    file.write("R Square,MAE,Train Time\n")

            with open(file_path, 'a') as file:
                file.write(f"{r_square},{mae},{train_time}\n")

            ## Cleanup
            del model
            K.clear_session()