from app.GeneticAlgorithm import GeneticAlgorithm
from app.get_indicators import get_select_indicator_values
from .DataPreprocesser import DataPreprocesser
import os
from .Chromosome import Limit, Chromosome
import tensorflow as tf
from app.Model import Model
from app.parameters import Architecture
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def create_tf_session():
    ## Only allocate required GPU space
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    tf.compat.v1.disable_eager_execution()


def optimise_params(symbol: str, should_use_indicators: bool):

    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto/{symbol}.parquet",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        sequence_length=50
    )
    if not preprocessor.has_loaded and should_use_indicators:
        indicator_df = get_select_indicator_values(preprocessor.df_original)
        preprocessor.change_data(indicator_df)
        preprocessor.print_df()

    preprocessor.preprocess()

    train_x, train_y = preprocessor.get_train()
    validation_x, validation_y = preprocessor.get_validation()



    ## Limits are inclusive
    limits = {
        "hidden_layers": Limit(1, 4),
        "neurons_per_layer": Limit(16, 128),
        "dropout": Limit(0.0, 0.5),
        "initial_learn_rate": Limit(0.000001, 0.1),
        "batch_size": Limit(50, 2000),
    }

    # Returns fitness for the specified chromosome
    def maximisation_fitness_func(chromosome: Chromosome) -> float:
        fitness = 0.0
        for key, value in chromosome.values.items():
            fitness += value
        return fitness
    


    def fitness_func(chromosome: Chromosome) -> float:
        create_tf_session()
        params = chromosome.values

        print("Current Chromosome Params:")
        print(params)
        model = Model(train_x, train_y, validation_x, validation_y,
            preprocessor.get_seq_info_str(),
            architecture=Architecture.GRU.value,
            is_bidirectional=True,
            batch_size=round(params["batch_size"]),
            hidden_layers=round(params["hidden_layers"]),
            neurons_per_layer=round(params["neurons_per_layer"]),
            dropout=params["dropout"],
            initial_learn_rate=params["initial_learn_rate"],
        )
        model.train()
        r_square = model.score["RSquaredMetric"]
        fitness = r_square
        chromosome.other["mae"] = model.score["mae"]

        ## Cleanup
        del model
        K.clear_session()

        return fitness


    ga = GeneticAlgorithm(limits, fitness_func,
        population_size=10, mutation_rate=0.025, generations=20,
        log_file="results/params_optimisation.csv")
    ga.start()

    ## Save best model to a specific folder

    plt.plot(ga.best_fitnesses)
    plt.title("Best Fitnesses over Epochs")
    plt.ylabel("Fitness")
    plt.xlabel("Epoch")
    plt.show()
