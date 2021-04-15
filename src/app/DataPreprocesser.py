from typing import Deque, List, Tuple
from numpy.core.numeric import NaN
import numpy as np
import pandas as pd
from collections import deque
import os
import random

import ta


_file_names = ["train_x.npy", "train_y.npy", "validation_x.npy", "validation_y.npy", "test_x.npy", "test_y.npy"]

def standardise(arr: np.array):
    return (arr - np.mean(arr)) / np.std(arr)

class DataPreprocesser():
    def __init__(self, dataset_file: str, col_names: List[str], forecast_col_name:str, *,
        max_dataset_size = 100000, forecast_period = 1, sequence_length = 100,
        test_split = 0.2, val_split = 0.2,
        should_ask_load = True, is_classification=False):
        """
        INFO GOES HERE
        """
        ## Param member vars
        self.is_classification=is_classification
        self.forecast_period = forecast_period
        self.max_dataset_size = max_dataset_size
        self.sequence_length = sequence_length
        self.dataset_folder, self.dataset_file = os.path.split(dataset_file)
        self.dataset_name = self.dataset_file.split(".")[0]
        self.forecast_col_name = forecast_col_name
        self.test_split = test_split
        self.val_split = val_split
        self.col_names = col_names
        self.state_folder = f"{os.environ['WORKSPACE']}/state/{self.get_seq_info_str()}"

        ## Other Member vars
        self.df = pd.DataFrame()
        self.df_original = pd.DataFrame() # Original df, no standardisation
        self.df_no_std = pd.DataFrame() # Original df, but with pct_change applied

        self.train_x, self.train_y = np.array([]), np.array([]) 
        self.validation_x, self.validation_y = np.array([]), np.array([]) 
        self.test_x, self.test_y = np.array([]), np.array([]) 

        self.has_loaded = False
        
        if self.state_folder is not None and not os.path.exists(self.state_folder):
            os.makedirs(self.state_folder)
        
        ### Check for existing training data
        if should_ask_load and self._is_existing_data() and self._should_use_existing_data():
            self._load_existing_data() # Loads sequences
        else:
            self._generate_df()



    def _balance_sequences(self, sequences_x: np.ndarray, sequences_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        target_index = list(self.df.columns).index("target")
        one_indexes, zero_indexes = [], []
        for index, target in enumerate(sequences_y):
            if target == 1: one_indexes.append(index)
            else: zero_indexes.append(index)

        remove_arr = one_indexes if len(one_indexes) > len(zero_indexes) else zero_indexes
        dif = abs(len(one_indexes) - len(zero_indexes))

        print(f"Num 1s: {len(one_indexes)}")
        print(f"Num 0s: {len(zero_indexes)}")
        print(f"Len x: {len(sequences_x)} --- Len y: {len(sequences_y)}")

        random.shuffle(remove_arr) # Shuffle removal order
        for i in range(dif):
            row = remove_arr.pop()
            sequences_y[row] = NaN
            for j in range(len(sequences_x[row])):
                sequences_x[row][j][0] = NaN
            
        sequences_x = sequences_x[~np.isnan(sequences_x).any(axis=2)].reshape(-1, sequences_x.shape[1], sequences_x.shape[2])
        sequences_y = sequences_y[~np.isnan(sequences_y)]

        one_indexes, zero_indexes = [], []
        for index, target in enumerate(sequences_y):
            if target == 1: one_indexes.append(index)
            else: zero_indexes.append(index)

        print(f"AFTER Num 1s: {len(one_indexes)}")
        print(f"AFTER Num 0s: {len(zero_indexes)}")
        print(f"AFTER Len x: {len(sequences_x)} --- Len y: {len(sequences_y)}")


        return sequences_x, sequences_y



    def _shuffle_seq(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        x = np.array(x)
        y = np.array(y)
        return x, y


    ### PUBLIC FUNCTIONS ###

    def preprocess(self): # This must be done from outside this class
        if self.has_loaded: # We have already preprocessed the data!
            print("\nWARNING: The data loaded was already preprocessed! Skipping the preprocessing step!\n")
            return
        ## Split validation and training set
        sequences_x, sequences_y = self._make_sequences(self.df, should_shuffle=False)
        if self.is_classification:
            sequences_x, sequences_y = self._balance_sequences(sequences_x, sequences_y)

        train_and_val_x, self.test_x = np.split(sequences_x, [-int(self.test_split * len(sequences_x))])
        train_and_val_y, self.test_y = np.split(sequences_y, [-int(self.test_split * len(sequences_y))])

        # train_and_val_x, train_and_val_y = self._shuffle_seq(train_and_val_x, train_and_val_y)
        
        self.train_x, self.validation_x = np.split(train_and_val_x, [-int(self.val_split * len(sequences_x))])
        self.train_y, self.validation_y = np.split(train_and_val_y, [-int(self.val_split * len(sequences_y))])

        self.train_x, self.train_y = self._shuffle_seq(self.train_x, self.train_y)
        self.validation_x, self.validation_y = self._shuffle_seq(self.validation_x, self.validation_y )


        ## Save sequences to npy files
        self.save_datasets()
        self.print_df()
        self.print_df_no_std()


    def print_dataset_totals(self):
        print(f"Training total: {len(self.train_y)}")
        print(f"Validation total: {len(self.validation_y)}")
        print(f"Test total: {len(self.test_y)}")

    def get_seq_info_str(self):
        if self.dataset_name is None:
            self.dataset_name = os.listdir(self.dataset_folder)[0]
        return f"{str(self.dataset_name)}-{'Class' if self.is_classification else 'Regress'}-SeqLen{self.sequence_length}-Forward{self.forecast_period}"

    def get_datasets(self):
        return self.train_x, self.train_y, self.validation_x, self.validation_y, self.test_x, self.test_y
    
    def get_train(self):
        return self.train_x, self.train_y
    
    def get_validation(self):
        return self.validation_x, self.validation_y

    def get_test(self):
        return self.test_x, self.test_y

    def get_df(self):
        return self.df

    def get_df_original(self):
        return self.df_original

    def get_df_no_std(self):
        return self.df_no_std

    def change_data(self, data: pd.DataFrame):
        """
        Changes the original df data then applies pct change then preprocesses it (pct_change and standardise)
        """
        self.df = data
        self._preprocess_df()
        




    def print_df(self):
        print(f"\n\nMAIN DF FOR PREDICTING: {self.forecast_col_name}")
        print(self.df.head(15))
        print("Columns:")
        print(self.df.columns)

    def print_df_no_std(self):
        print(f"\n\nDF NO STANDARDISATION")
        print(self.df_no_std.head(15))
        print("Columns:")
        print(self.df.columns)


    def save_datasets(self):
        np.save(f"{self.state_folder}/{_file_names[0]}", self.train_x)
        np.save(f"{self.state_folder}/{_file_names[1]}", self.train_y)
        np.save(f"{self.state_folder}/{_file_names[2]}", self.validation_x)
        np.save(f"{self.state_folder}/{_file_names[3]}", self.validation_y)
        np.save(f"{self.state_folder}/{_file_names[4]}", self.test_x)
        np.save(f"{self.state_folder}/{_file_names[5]}", self.test_y)


    ### PRIVATE FUNCTIONS ###    
        

    def _is_existing_data(self):
        dir_items = os.listdir(self.state_folder)
        ret = all([x in dir_items for x in _file_names])
        if ret:
            print("\n\nFound existing training and validation data.")
        return ret 

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
        self.train_x = np.load(f"{self.state_folder}/{_file_names[0]}")
        self.train_y = np.load(f"{self.state_folder}/{_file_names[1]}")
        self.validation_x = np.load(f"{self.state_folder}/{_file_names[2]}")
        self.validation_y = np.load(f"{self.state_folder}/{_file_names[3]}")
        self.test_x = np.load(f"{self.state_folder}/{_file_names[3]}")
        self.test_y = np.load(f"{self.state_folder}/{_file_names[4]}")
        self.has_loaded = True

    def _generate_df(self):
        print("Generating new arrays of sequences for training...")
        self.df = self._get_main_dataframe()
        self.print_df()
        self.print_df_no_std()
        self._preprocess_df()    


    def _relative_change(self, df: pd.DataFrame):
        for col in df.columns:
            avg_period = 10
            avgs = ta.trend.SMAIndicator(df[col], avg_period).sma_indicator()
            df[col] = df[col].sub(avgs).div(avgs)
        df.dropna(inplace=True)
        return df


    def _relative_change_col(self, df: pd.DataFrame, col: str):
        avg_period = 10
        avgs = ta.trend.SMAIndicator(df[col], avg_period).sma_indicator()
        df[col] = df[col].sub(avgs).div(avgs)
        df.dropna(inplace=True)
        return df

    def _pct_chg(self, df: pd.DataFrame):
        for col in df.columns:
            if np.sum(df[col].to_numpy() == 0) > 0:
                df[col] = df[col] + 1 # Avoid zero errors
            
            df[col] = df[col].pct_change()
            df[col].replace([np.inf, -np.inf], NaN, inplace=True)
            df.dropna(inplace=True)
        return df

    def _standardise(self, df: pd.DataFrame):
        ##### STANDARDISATION #####
        ## Normalise all data (except target price)
        for col in df.columns:
            if col != "target": # Don't normalise the target!
                df[col] = standardise(df[col].values)
        df.dropna(inplace=True)
        return df


    def _preprocess_df(self):
        """
        Add target and standardise data in df
        """
        self.df_original = self.df.copy()

        ## Turn into pct change
        self.df = self._pct_chg(self.df)

        self._add_target()
        self.df_original["target"] = self.df["target"]
        self.df_no_std = self.df.copy()

        self.df = self._standardise(self.df)

    def _add_target(self):
        """
        When adding the target, the df columns must be in percent change, and not yet normalised
        """
        ## Add future price column to main_df (which is now the target)
        future = []

        values = self.df[self.forecast_col_name]

        for i in range(len(values)):
            if i >= len(values) - self.forecast_period:
                future.append(NaN) # We can't forecast these so we remove them
                continue
            combined_pct = 1
            for x in values[i + 1:i + self.forecast_period + 1]:
                combined_pct *= (1 + x)
            combined_pct -= 1

            if self.is_classification:
                number = 1 if combined_pct > 0 else 0
                future.append(number)
            else:
                future.append(combined_pct) # Add the sum of the last x % changes

        self.df["target"] = future
        self.df.dropna(inplace=True)


    


    def _load_df(self, path: str, *, date_column = "time"):
        if path.endswith(".parquet"):
            df = pd.read_parquet(path) # Index is automatically set to open_time
        else:
            df = pd.read_csv(path, parse_dates=[date_column])
        return df


    def _get_main_dataframe(self):  
        path = f"{self.dataset_folder}/{self.dataset_file}"
        df = pd.DataFrame()
        df = self._load_df(path)
        df = df[-self.max_dataset_size:] # Reduce dataset size to max size
        df = df[self.col_names]
       

        df.dropna(inplace=True)
        return df


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