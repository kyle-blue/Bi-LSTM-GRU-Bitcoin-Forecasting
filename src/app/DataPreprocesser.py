from typing import Deque, List
from numpy.core.numeric import NaN
import numpy as np
import pandas as pd
from collections import deque
import os
import random


__file_names = ["train_x.npy", "train_y.npy", "validation_x.npy", "validation_y.npy", "test_x.npy", "test_y.npy"]

def standardise(arr: np.array):
    return (arr - np.mean(arr)) / np.std(arr)

class DataPreprocesser():
    def __init__(self, dataset_folder: str, col_names: List[str], forecast_col_name:str, *,
        max_dataset_size = 100000, forecast_period = 1, sequence_length = 100,
        state_folder: str = None, test_split = 0.2, val_split = 0.2, forecast_file: str = None):
        """
        """
        ## Param member vars
        self.forecast_period = forecast_period
        self.max_dataset_size = max_dataset_size
        self.sequence_length = sequence_length
        self.state_folder = state_folder
        self.forecast_file = forecast_file
        self.forecast_col_name = forecast_col_name
        self.test_split = test_split
        self.val_split = val_split
        self.dataset_folder = dataset_folder
        self.col_names = col_names

        ## Other Member vars
        self.df = pd.DataFrame()
        self.df_original = pd.DataFrame() # Original df, no standardisation
        self.df_no_std = pd.DataFrame() # Original df, but with pct_change applied

        self.train_x, self.train_y = np.array([]), np.array([]) 
        self.validation_x, self.validation_y = np.array([]), np.array([]) 
        self.test_x, self.test_y = np.array([]), np.array([]) 

        if state_folder is not None and not os.path.exists(state_folder):
            os.makedirs(state_folder)
        
        ### Check for existing training data
        if self._is_existing_data() and self._should_use_existing_data():
            self._load_existing_data() # Loads sequences
        else:
            self._generate_df()


    ### PUBLIC FUNCTIONS ###

    def generate_datasets_from_df(self): # This must be done from outside this class
        ## Split validation and training set
        train_and_val_df, test_df = np.split(self.df, [-int(self.test_split * len(self.df))])
        train_and_val_df.sample(frac = 1) # Shuffle validation and train together (but not test)
        train_df, validation_df = np.split(train_and_val_df, [-int(self.val_split * len(self.df))]) # Validation is same size as test_df

        ## Make sequences
        self.train_x, self.train_y = self._make_sequences(train_df)
        self.validation_x, self.validation_y = self._make_sequences(validation_df)
        self.test_x, self.test_y = self._make_sequences(test_df, should_shuffle = False)

        ## Save sequences to npy files
        self.save_datasets()
        self.print_df()    


    def get_df(self):
        return self.df

    def get_df_original(self):
        return self.df_original

    def get_df_no_std(self):
        return self.df_no_std

    def add_data(self, data: pd.DataFrame):
        """
        Adds additional data to the main dataframe
        """
        self.df.join(data, how="outer")
        self.df.dropna(inplace=True)

    def print_df(self):
        print(f"\n\nMAIN DF FOR PREDICTING: {self.forecast_col_name}")
        print(self.df.head(15))

    def save_datasets(self):
        np.save(f"{self.state_folder}/{__file_names[0]}", self.train_x)
        np.save(f"{self.state_folder}/{__file_names[1]}", self.train_y)
        np.save(f"{self.state_folder}/{__file_names[2]}", self.validation_x)
        np.save(f"{self.state_folder}/{__file_names[3]}", self.validation_y)
        np.save(f"{self.state_folder}/{__file_names[4]}", self.test_x)
        np.save(f"{self.state_folder}/{__file_names[5]}", self.test_y)


    ### PRIVATE FUNCTIONS ###    
        

    def _is_existing_data(self):
        dir_items = os.listdir(self.state_folder)
        print("\n\nFound an training and validation data.")
        return all([x in dir_items for x in __file_names])

    def _should_use_existing_data(self):
        """
        Asks the user whether they would like to use existing data
        """
        print("\nPlease select an option:")
        print("1. Use existing data")
        print("2. Generate new data")
        while True:
            user_input = int(input())
            if user_input == 1:
                return True
            if user_input == 2:
                return False

    
    
    def _load_existing_data(self):
        print("Using existing data...")
        self.train_x = np.load(f"{self.state_folder}/{__file_names[0]}")
        self.train_y = np.load(f"{self.state_folder}/{__file_names[1]}")
        self.validation_x = np.load(f"{self.state_folder}/{__file_names[2]}")
        self.validation_y = np.load(f"{self.state_folder}/{__file_names[3]}")
        self.test_x = np.load(f"{self.state_folder}/{__file_names[3]}")
        self.test_y = np.load(f"{self.state_folder}/{__file_names[4]}")

    def _generate_df(self):
        print("Generating new arrays of sequences for training...")
        self.df = self._get_main_dataframe()
        self.print_df()
        self.df = self._preprocess_df()    


    


    def _preprocess_df(self):
        """
        Add target and standardise data in df
        """
        
        self.df_original = self.df.copy()

        ## Turn into pct change
        for col in self.df.columns:
            if str(col).endswith("_volume"):
                self.df[col] = self.df[col] + 1 # Avoid zero errors
            self.df[col] = self.df[col].pct_change()
            self.df[col].replace([np.inf, -np.inf], NaN, inplace=True)
            self.df.dropna(inplace=True)

        self._add_target()
        self.df_original["target"] = self.df["target"]

        self.df_no_std = self.df.copy()

        ##### STANDARDISATION #####
        ## Normalise all data (except target price)
        for col in self.df.columns:
            if col != "target": # Don't normalise the target!
                self.df[col] = standardise(self.df[col].values)
        self.df.dropna(inplace=True)
        

    def _add_target(self):
        """
        When adding the target, the df columns must be in percent change, and not yet normalised
        """
        ## Add future price column to main_df (which is now the target)
        future = []

        symbol_data = pd.DataFrame()
        is_multiple_dataset_files = len(set(os.listdir(self.dataset_folder))) > 1
        if is_multiple_dataset_files: 
            symbol_data = self.df[f"{self.forecast_file}_{self.forecast_col_name}"]
        else:
            symbol_data = self.df[self.forecast_col_name]
        symbol_data_len = len(symbol_data)

        for i in range(symbol_data_len):
            if i >= symbol_data_len - self.forecast_period:
                future.append(NaN) # We can't forecast these so we remove them
                continue
            future.append(sum(symbol_data[i:i + self.forecast_period])) # Add the sum of the last x % changes

        self.df["target"] = future
        self.df.dropna(inplace=True)


    


    def _load_df(self, path: str, *, date_column = "time"):
        if path.endswith(".parquet"):
            df = pd.read_parquet(path) # Index is automatically set to open_time
        else:
            df = pd.read_csv(path, parse_dates=[date_column])
        return df


    def _get_main_dataframe(self):
        is_multiple_dataset_files = len(set(os.listdir(self.dataset_folder))) > 1
        if not is_multiple_dataset_files:
            df = self._load_df(os.listdir(self.dataset_folder)[0])
            df = df[-self.max_dataset_size:] # Reduce dataset size to max size
            df.dropna(inplace=True)
            return df

        ## Else there are multiple dataset files        
        main_df = pd.DataFrame()
        files = set(os.listdir(self.dataset_folder)) # Remove file extension from strings
        for file in files:
            file_name =  file.split(".")[0] # Remove file extension from file
            df = pd.DataFrame()
            df = self._load_df(file)
            df = df[self.col_names]

            rename_dict = {}
            for col_name in self.col_names:
                rename_dict[col_name] = f"{file_name}_{col_name}"
            df.rename(columns=rename_dict, inplace=True)

            df = df[-self.max_dataset_size:] # Reduce dataset size to max size

            if len(main_df) == 0: main_df = df
            else: main_df = main_df.join(df, how="outer")

            main_df.dropna(inplace=True)
        return main_df


    ## returns train_x and train_y
    def _make_sequences(self, df, *, should_shuffle = True):
        ## Create sequences
        # [
        #    [[sequence1], target1]
        #    [[sequence2], target2]
        # ]        
        
        ## MAKE INTO SEQUENCES
        sequences: list = [] 
        cur_sequence: Deque = deque(maxlen=self.sequence_length)
        target_index = df.columns.get_loc("target")
        for index, value in enumerate(df.to_numpy()):
            # Since value is only considered a single value in the sequence (even though itself is an array), to make it a sequence, we encapsulate it in an array so:
            # sequence1 = [[values1], [values2], [values3]]
            cur_sequence.append(value[:target_index]) # Append all but target to cur_sequence
            if len(cur_sequence) == self.sequence_length:
                seq = list(cur_sequence)
                sequences.append([np.array(seq), value[target_index]]) # value[-1] is the target
        df.drop_duplicates(inplace=True)
        
        
        if should_shuffle:
            random.shuffle(sequences) # Shuffle sequences to avoid order effects on learning

        data_x = []
        data_y = []
        for seq, target in sequences:
            data_x.append(seq)
            data_y.append(target)
        
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        return data_x, data_y