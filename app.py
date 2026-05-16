import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model + vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("📰 AI-Powered Misinformation Detection System")

st.markdown("Enter a news article or headline below to check whether it is REAL or FAKE.")

news = st.text_area("Enter News Text", height=250)

if st.button("Analyze"):

    if news.strip() == "":
        st.warning("Please enter some text.")

    else:
        cleaned = clean_text(news)
        vector_input = vectorizer.transform([cleaned])

        prediction = model.predict(vector_input)

        # ✅ REAL PROBABILITY FIXED (NO ERROR NOW)
        probability = model.predict_proba(vector_input)

        real_probability = probability[0][0] * 100
        fake_probability = probability[0][1] * 100
        confidence = max(real_probability, fake_probability)

        st.subheader("Result")

        if prediction[0] == 1:
            st.error("🚨 FAKE NEWS DETECTED")
        else:
            st.success("✅ REAL NEWS DETECTED")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.write("### Probabilities")
        st.write(f"Real News: {real_probability:.2f}%")
        st.progress(int(real_probability))

        st.write(f"Fake News: {fake_probability:.2f}%")
        st.progress(int(fake_probability))

st.markdown("---")
st.caption("Built using ML + NLP + Streamlit")