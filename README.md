# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the weather dataset and handle missing values by replacing them with the mean of their respective columns.
2.Feature Selection: Define the input features (X) using variables like humidity and CO_2, and set the target variables (y) for Pollution, Temperature, and Energy.
3.Model Training: Split the data into training and testing sets (80/20) and train a DecisionTreeRegressor with a maximum depth of 5 for each target.
4.Evaluation: Predict values on the test set and calculate performance metrics, specifically RMSE for error margin and R^2 Score for accuracy.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Sridharan B
RegisterNumber:  212225230272
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)  # Automatically fits and assigns clusters


plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(X[df['Cluster']==i]['Annual Income (k$)'],
                X[df['Cluster']==i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids', marker='X')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


print(df)
```

## Output:

<img width="877" height="803" alt="Screenshot 2026-05-15 144856" src="https://github.com/user-attachments/assets/50c51e88-a6a5-411b-b32b-4a96c05be0ff" />


## Result:
