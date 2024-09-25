import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Add your data preprocessing logic here
    return data
