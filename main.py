from src.data_preprocessing import basic_preprocessing
from src.feature_engineering import feature_engineering
from src.model import train_model
import pandas as pd

data = pd.read_csv("Data/Raw/train.csv")
data = basic_preprocessing(data)
data = feature_engineering(data)
train_model(data)