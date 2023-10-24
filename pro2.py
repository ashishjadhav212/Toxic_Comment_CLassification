import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open('classification_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Toxic Comment Classification")

# User input
user_input = st.text_area("Enter a comment:")

# Create a button for making predictions
if st.button("Check Toxicity"):
    if user_input:
        
        user_input_tfidf = tfidf_vectorizer.transform([user_input])

       
        prediction = model.predict(user_input_tfidf)[0]

        # Display the prediction
        if prediction == 0:
            st.success("This comment is non-toxic.")
        else:
            st.error("This comment is toxic.")


