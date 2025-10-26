# Import all the required libraries

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# % matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import joblib

# Load the data

ipl_data = pd.DataFrame({
    'team1_wins_last_5': [3, 4, 2, 5, 1],
    'team2_wins_last_5': [2, 1, 4, 3, 4],
    'venue_advantage': [1, 0, 0, 1, 0],
    'toss_winner': [1, 0, 1, 1, 0],
    'winner': ['team1', 'team2', 'team2', 'team1', 'team2']
})


# Split and Train the model

features = ['team1_wins_last_5', 'team2_wins_last_5', 'venue_advantage', 'toss_winner']

X = ipl_data[[features]]
y = ipl_data[['winner']]

model = RandomForestClassifier()

model.fit(X,y)


# Predictions

def predict(team1_form, team2_form, venue, toss):
    prediction = model.predict(X)
    confidence = model.predict_proba(X)
    return f'winner: {prediction[0]} (Confidence: {confidence: .2%})'


# joblib is used to save the .py to .pkl file

joblib.dump(model, 'model.pkl')

# print('Model successfully saved as model.pkl')