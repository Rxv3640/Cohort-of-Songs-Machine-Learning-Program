import pandas as pd

data = pd.read_csv("rolling_stones_spotify.csv")

#Initial data inspection and data cleaning

check_null = data.isnull().sum()

check_duplicates = data.duplicated().any()

print("Number of null values\n")
print(check_null, '\n')

print("True for there are duplicates, and False if there are not any duplicates\n")
print(check_duplicates, '\n')

#Perform exploratory data analysis

import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = data.select_dtypes(include='number').corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Examining relationship between a song's popularity and various factors")
plt.show()

number_of_names = data['name'].value_counts()

names = data['name'].unique()

number_of_names.head(20).plot(kind='bar')
plt.xlabel('Song names')
plt.ylabel('Number of popular songs')
plt.title('Identify the two albums with the most popular songs.')
plt.tight_layout()
plt.show()