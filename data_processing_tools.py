# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/aqibjallal/Downloads/Study Material/ML:AI/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 Part 1 - Data Preprocessing/Python/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

