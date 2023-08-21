# Code for training the model that will be deployed
# It saves the model with pickle
# Soogeun 21082023 #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('hiring.csv')

print(dataset.head())

# for the experience column,
# filling the missing values (Nan) by 0:
dataset['experience'].fillna(0, inplace = True)

# for the test_score column, fill the missing values by the average:
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace = True)

X = dataset.iloc[:, :3]

# converting the words in the experience column to integers:
def to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

# applying the function to X:
X['experience'] = X['experience'].apply(lambda x: to_int(x))

print(X)

# outcome feature is salary:
y = dataset.iloc[:, -1]

regressor = LinearRegression()

# fitting the model
regressor.fit(X, y)

# saving the model to disk:
pickle.dump(regressor, open('salary_model.pkl', 'wb'))

# Loading model to compare the results
saved_model = pickle.load(open('salary_model.pkl','rb'))
print(saved_model.predict([[2, 9, 6]]))
