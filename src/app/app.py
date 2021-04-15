import os
from app.DataPreprocesser import DataPreprocesser
from app.Model import Model
from app.get_indicators import get_select_indicator_values
from app.multi_test import multi_test
from app.optimise_params import create_tf_session, optimise_params
from app.parameters import Architecture, Symbol
from app.test_model import test_model
from .indicator_correlations import indicator_correlations

SYMBOL_TO_PREDICT = Symbol.BTC_USDT.value
SHOULD_USE_INDICATORS = False
IS_CLASSIFICATION = True

def start():
    create_tf_session()
    

    print("\n\n\n")
    print("Please choose an option:")
    print("1. Train a new model")
    print("2. Test an existing model")
    print("3. Test indicator correlations")
    print("4. Optimise RNN params using Genetic Algorithm")
    print("5. Multi Test")
    is_valid = False
    while not is_valid:
        inp = int(input())
        if inp == 1:
            train_model()
            is_valid = True
        if inp == 2:
            test_model()
            is_valid = True
        if inp == 3:
            indicator_correlations(SYMBOL_TO_PREDICT)
            is_valid = True
        if inp == 4:
            optimise_params(SYMBOL_TO_PREDICT, SHOULD_USE_INDICATORS)
            is_valid = True
        if inp == 5:
            multi_test(SYMBOL_TO_PREDICT)
            is_valid = True
        if not is_valid:
            print("Please choose a valid option...")

    

def train_model():
    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto/{SYMBOL_TO_PREDICT}.parquet",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        sequence_length=100,
        forecast_period=10,
        is_classification=IS_CLASSIFICATION
    )
    if not preprocessor.has_loaded and SHOULD_USE_INDICATORS:
        indicator_df = get_select_indicator_values(preprocessor.df_original)
        preprocessor.change_data(indicator_df)
        preprocessor.print_df()
        preprocessor.print_df_no_std()


    preprocessor.preprocess()

    train_x, train_y = preprocessor.get_train()
    validation_x, validation_y = preprocessor.get_validation()

    model = Model(
        train_x, train_y, validation_x, validation_y,
        preprocessor.get_seq_info_str(),
        architecture=Architecture.GRU.value,
        is_bidirectional=False,
        batch_size=1534,
        hidden_layers=2,
        neurons_per_layer=60,
        dropout=0.31,
        initial_learn_rate=0.0037,
        early_stop_patience=6,
        max_epochs=100,
        is_classification=IS_CLASSIFICATION
    )
    
    preprocessor.print_dataset_totals()
    del preprocessor # Save memory

    model.train()
    model.save_model()



