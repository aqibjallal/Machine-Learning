# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handling missing data
from sklearn.impute import SimpleImputer

Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Imputer.fit(x[:, 1:3])
x[:, 1:3] = Imputer.transform(x[:, 1:3])
print(x)
