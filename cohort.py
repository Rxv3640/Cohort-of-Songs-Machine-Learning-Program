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

#Perform cluster analysis

from sklearn.cluster import KMeans
import numpy as np

X = data[['energy', 'instrumentalness']]

wcss = []

for i in range(1,11):
  model = KMeans(n_clusters=i, n_init=10, init='k-means++', random_state=42)
  model.fit(X)
  wcss.append(model.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Finding optimal number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

model = KMeans(n_clusters = 3, n_init = 10, init = 'k-means++', random_state=42)
y_kmeans = model.fit_predict(X)

X = X.values

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.title('Cohorts of songs')
plt.xlabel('Energy')
plt.ylabel('Instrumentalness')
plt.legend()
plt.show()