# Description : This program detects if someone has diabetes using machine learning and python !

#Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Create a title and a sub title
st.write("""
# Diabetes detection
Detect if someone has diabetes using machine learning and python !
""")

# Display and image
image = Image.open('C:/Users/Serry/Desktop/Final Project/detection.png')
st.image(image , caption = 'Machine Learning' , use_column_width=True)

# Get the data
df = pd.read_csv('C:/Users/Serry/Desktop/Final Project/diabetes.csv')

# Set a subheader
st.subheader('Data Information: ')

# Show data as a table
st.dataframe(df)

# Show statistics on the data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:,0:8].values
Y = df.iloc[:,-1].values

# Split the dataset into 75% Training and 25% Testing
X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.25, random_state=0)

# Get the feature iput from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('blood_pressure',0,122,72)
    skin_thickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('insulin',0.0,846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    dbf = st.sidebar.slider('dbf',0.078,2.42,0.3725)
    age = st.sidebar.slider('age',21,81,29)
    
    # Store a dictionary in variable
    user_data = {'pregnancies':pregnancies,
                 'glucose' :glucose,
                 'blood_pressure':blood_pressure,
                 'skin_thickness':skin_thickness,
                 'insulin':insulin,
                 'BMI':BMI,
                 'dbf':dbf,
                 'age':age
                 }
    
    # Transofrm the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the user input
st.subheader('User Input: ')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train , Y_train)

# Show the model's metrics
st.subheader('Model Test Accurace Score')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

# Store the model's prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)



     
    




