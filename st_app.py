import streamlit
import joblib
import warnings
import numpy as np
warnings.filterwarnings('ignore')


streamlit.title('IPL Prediction App')
streamlit.write('Helping betting babulu in voodagotting the money in betting')
streamlit.write('Batting with money in betting game')

#Sidebar for inputs
streamlit.sidebar.header('Enter ipl team')


# Input fields

team1_wins = streamlit.sidebar.slider('Team1 wins in last 5 matches', min_value = 0, max_value = 5, value = 0)
team2_wins = streamlit.sidebar.slider('Team2 wins in last 5 matches', min_value = 0, max_value = 5, value = 0)

# team1 = 1, team2 = 2
venue_adv = streamlit.sidebar.selectbox('Advantages of  venue', [1, 2])

# team1 = 1, team2 = 2
toss_won = streamlit.sidebar.selectbox('Toss won by team', [1, 2])


# Load the model
@streamlit.cache_data
def get_model():
    ipl_model = joblib.load("C:/Users/manik/TTT/MLOps/first_model/model.pkl")
    return ipl_model

# predict button
if streamlit.sidebar.button('Predict the winner'):
    ipl_winner = get_model()
    # print(f'Winner: {ipl_winner}')

    # # create feature vector
    features = np.array([[team1_wins, team2_wins, venue_adv, toss_won]])

    # Prediction
    prediction = ipl_winner.predict(features)

    #confidence score
    confidence_probs = ipl_winner.predict_proba(features)
    confidence_score = np.max(confidence_probs) * 100 # Highest probability

    streamlit.write(f'Predicted result: {prediction}')

# Information section
with streamlit.expander('How it works'):
    streamlit.write('''
             This app used Machine Learning to predict the ipl winner based on:
             - **Last five wins** of team1 and team2
             - **Venue advantage** of team1 and team2
             - **Toss won** between team1 and team2
             ''')
     

streamlit.markdown('---')
streamlit.markdown('Thanks to betting babulu')