import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

df = pd.read_csv('spam_new.csv',encoding= 'latin-1')

# X will be the features
X = np.array(df["message"])

# y will be the target variable
y = np.array(df["class"])
cv = CountVectorizer()

X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.33,
                                                random_state=42)

model = BernoulliNB()
model.fit(X_train, y_train)
# Function for model prediction
def model_prediction(features):
    features = cv.transform([features]).toarray()
    Message = str(list(model.predict(features)))
    
    return Message

def app_design():

    text= st.text_area("Enter your text")
    # Create a feature list from the user inputs
    features = text # add features according to notebook
    
    # Make a prediction when the user clicks the "Predict" button
    if st.button('Predict Spam'):  
        predicted_value = model_prediction(features)
        if predicted_value == "['ham']":
            st.success("Your comment is not spam") 
        elif predicted_value == "['spam']":
            st.success("Your Comment is spam")      

def main():

        # Set the app title and add your website name and logo
        st.set_page_config(
        page_title="Spam Detection",
        page_icon=":chart_with_upwards_trend:",
        )
    
        st.title("Welcome to our Spam Detection App!")
    
        app_design()    


if __name__ == '__main__':
        main()