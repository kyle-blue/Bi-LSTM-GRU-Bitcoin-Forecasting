import pandas as pd
import os



def start():
    data_folder = os.environ["WORKSPACE"] + "/data"
    input_folder = data_folder + "/parquets"
    output_folder = data_folder + "/crypto"

    inputs = os.listdir(input_folder)
    for file_name in inputs:
        input_file = f"{input_folder}/{file_name}"
        output_file = f"{output_folder}/{file_name.split('.')[0]}.csv"
        print(f"\n{file_name}:")
        dataset = pd.read_parquet(input_file)
        print(dataset)
        dataset.head(20)