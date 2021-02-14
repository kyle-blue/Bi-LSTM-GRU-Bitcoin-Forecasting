import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Sequential
import os

def test_model():
    model_dir = f'{os.environ["WORKSPACE"]}/models/final'
    dir_items = os.listdir(model_dir)
    dir_items.remove(".temp")
    print("Please choose an model to test from the list (in dir ./model_config):")

    for index, item in enumerate(dir_items):
        print(f"{index}. {item}")

    is_valid_input = False
    chosen_file = ""
    while not is_valid_input:
        user_input = int(input())
        if user_input < len(dir_items):
            is_valid_input = True
        chosen_file = dir_items[user_input]
        if not is_valid_input:
            print("Please enter a valid input")

    chosen_path = f"{model_dir}/{chosen_file}"
    
    model_config_path = f'{os.environ["WORKSPACE"]}/model_config/model_config.json'
    model_config = "" 
    with open(model_config_path, 'r') as file:
        model_config = file.read()
    model: Sequential = keras.models.model_from_json(model_config)
    model.load_weights(chosen_path)
    print(f"Successfully loaded model config from {model_config_path} and weights from {chosen_path}")
    
