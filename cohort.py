import pandas as pd

data = pd.read_csv("rolling_stones_spotify.csv")

#Initial data inspection and data cleaning

check_null = data.isnull().sum()

check_duplicates = data.duplicated().any()

print("Number of null values\n")
print(check_null, '\n')

print("True for there are duplicates, and False if there are not any duplicates\n")
print(check_duplicates, '\n')

#Perform exploratory data analysis and feature engineering

