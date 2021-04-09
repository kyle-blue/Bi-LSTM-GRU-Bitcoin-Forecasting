import math
import tensorflow as tf
import os

from tensorflow.python.ops.gen_batch_ops import batch
from app.Chromosome import Chromosome, Limit
from app.DataPreprocesser import DataPreprocesser
from app.GeneticAlgorithm import GeneticAlgorithm
from app.Model import Model
from app.parameters import Architecture, Symbol
from app.test_model import test_model
from .indicator_correlations import indicator_correlations
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow.keras.backend as K

SYMBOL_TO_PREDICT = Symbol.BTC_USDT.value

def create_tf_session():
    ## Only allocate required GPU space
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    tf.compat.v1.disable_eager_execution()


def start():
    create_tf_session()

    print("\n\n\n")
    print("Please choose an option:")
    print("1. Train a new model")
    print("2. Test an existing model")
    print("3. Test indicator correlations")
    print("4. Optimise RNN params using Genetic Algorithm")
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
            optimise_params()
            is_valid = True
        if not is_valid:
            print("Please choose a valid option...")

    

def train_model():
    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        forecast_file=f"{SYMBOL_TO_PREDICT}.parquet",
        sequence_length=250
    )
    preprocessor.preprocess()

    train_x, train_y = preprocessor.get_train()
    validation_x, validation_y = preprocessor.get_validation()

    model = Model(
        train_x, train_y, validation_x, validation_y,
        preprocessor.get_seq_info_str(),
        architecture=Architecture.LSTM.value,
        is_bidirectional=True,
        batch_size=2048,
        hidden_layers=4,
        neurons_per_layer=128,
    )
    
    preprocessor.print_dataset_totals()
    del preprocessor # Save memory

    model.train()
    model.save_model()


def set_session_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def optimise_params():
    random_seed = 12321
    set_session_seed(random_seed)


    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        forecast_file=f"{SYMBOL_TO_PREDICT}.parquet",
        sequence_length=250
    )
    preprocessor.preprocess()

    train_x, train_y = preprocessor.get_train()
    validation_x, validation_y = preprocessor.get_validation()



    ## Limits are inclusive
    limits = { 
        "batch_size": Limit(100, 2048),
        "hidden_layers": Limit(1, 4),
        "neurons_per_layer": Limit(16, 128),
        "dropout": Limit(0.0, 0.5),
        "initial_learn_rate": Limit(0.000001, 1.0)
    }

    # Returns fitness for the specified chromosome
    def maximisation_fitness_func(chromosome: Chromosome) -> float:
        fitness = 0.0
        for key, value in chromosome.values.items():
            fitness += value
        return fitness
    
    def fitness_func(chromosome: Chromosome) -> float:
        create_tf_session()
        set_session_seed(random_seed)

        params = chromosome.values
        model = Model(train_x, train_y, validation_x, validation_y,
            preprocessor.get_seq_info_str(),
            architecture=Architecture.LSTM.value,
            is_bidirectional=True,
            random_seed=random_seed,
            batch_size=round(params["batch_size"]),
            hidden_layers=round(params["hidden_layers"]),
            neurons_per_layer=round(params["neurons_per_layer"]),
            dropout=params["dropout"],
            initial_learn_rate=params["initial_learn_rate"]
        )
        model.train()
        val_mae = model.score["val_mae"]
        fitness = 1 / val_mae

        ## Cleanup
        del model
        K.clear_session()

        return fitness


    ga = GeneticAlgorithm(limits, fitness_func,
        population_size=100, mutation_rate=0.01, generations=100)
    ga.start()

    ## Save best model to a specific folder

    plt.plot(ga.best_fitnesses)
    plt.title("Best Fitnesses over Epochs")
    plt.ylabel("Fitness")
    plt.xlabel("Epoch")
    plt.show()



    # ind = ta.trend.MACD(df[f"{symbol}_close"], fillna=True)
    # df[f"{symbol}_macd_fast"] = ind.macd()
    # df[f"{symbol}_macd_signal"] = ind.macd_signal()
    # df[f"{symbol}_macd_histogram"] = ind.macd_diff()

    # ind = ta.momentum.RSIIndicator(df[f"{symbol}_close"], fillna=True)
    # df[f"{symbol}_rsi"] = ind.rsi()

    # ind = ta.trend.ADXIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
    # df[f"{symbol}_adx"] = ind.adx()
    # df[f"{symbol}_adx_neg"] = ind.adx_neg()
    # df[f"{symbol}_adx_pos"] = ind.adx_pos()

    # ind = ta.volume.AccDistIndexIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], df[f"{symbol}_volume"], fillna=True)
    # df[f"{symbol}_acc_dist"] = ind.acc_dist_index()

    # ind = ta.volatility.AverageTrueRange(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
    # df[f"{symbol}_atr"] = ind.average_true_range()
