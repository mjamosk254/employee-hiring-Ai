import streamlit as st
import tensorflow as tf 
import numpy as np

#load the saved Ai model
model=tf.keras.models.load_model("employee_hiring_model.keras")

#application title & paragraph
st.title("Employee Hiring Prediction System")
st.write("Enter employee details below.")

#input box for years of experience
experience=st.number_input("Years of Experience",min_value=0,max_value=50)

#input box for interview score
interview=st.number_input("Interview Score",min_value=0,max_value=100)

#input box for comms score
communication=st.number_input("Communication Score",min_value=0,max_value=100)

#predict button
if st.button("Predict"):
    #store the users input inside a numpy array
    employee=np.array([[experience,interview,communication]])
    #ask the Ai model to make a prediction
    prediction=model.predict(employee)
    #display the prediction probability
    st.write("prediction probability")
    st.write(prediction)

    #display final prediction
    if prediction[0][0]>=0.5:
        st.success("prediction: Hire")
    else:
        st.error("prediction: Reject")    