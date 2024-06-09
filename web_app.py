import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

port_stem = PorterStemmer()

# Load the vectorizer and model
vector_form = pickle.load(open("vector.pkl", "rb"))
load_model = pickle.load(open("model.pkl", "rb"))


def stemming(content):
    text_con = re.sub("[^a-zA-Z]", " ", content)
    text_con = text_con.lower()
    text_con = text_con.split()
    text_con = [
        port_stem.stem(word)
        for word in text_con
        if not word in stopwords.words("english")
    ]
    text_con = " ".join(text_con)
    return text_con


def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    predictions = load_model.predict(vector_form1)
    return predictions


if __name__ == "__main__":
    st.title("Fake News Detection App")
    st.subheader("Please enter your News content")
    sentence = st.text_area("Enter news content here", "", height=200)
    predict_btt = st.button("Predict")
    if predict_btt:
        prediction_class = fake_news(sentence)
        if prediction_class == [0]:
            st.success("This news is correct!")
        if prediction_class == [1]:
            st.warning("This news is NOT correct!")
