import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit app
st.title("AI vs Human Writing Detector")
st.write("This app predicts whether the given text is AI-generated or Human-written.")

# Text input
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Preprocess and predict
        features = vectorizer.transform([user_input])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("🤖 The text is AI Generated.")
        else:
            st.success("🧑 The text is Human Written.")
    else:
        st.warning("Please enter some text before predicting.")

