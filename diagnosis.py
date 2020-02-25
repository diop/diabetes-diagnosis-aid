# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X.isna().sum() # No null values



X.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
y.columns = ['Outcome']
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Exporing the Data
plt.figure(figsize=(18, 8))

for i, column in enumerate(X.columns):
    plt.subplot(2, 4, i+1)
    sns.distplot(X[column])
    plt.title(f"Distribution of {column}.")

plt.tight_layout()
plt.show()