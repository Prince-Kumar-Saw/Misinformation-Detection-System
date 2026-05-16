import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ---------------- NLP SETUP ----------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

h1, h2, h3 {
    color: white;
}

textarea {
    border-radius: 12px !important;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.real-news {
    background-color: rgba(34,197,94,0.2);
    border: 2px solid #22c55e;
}

.fake-news {
    background-color: rgba(239,68,68,0.2);
    border: 2px solid #ef4444;
}

.metric-card {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    margin-top: 10px;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 About This Project")

st.sidebar.info("""
This AI system detects whether a news article is REAL or FAKE using:

✅ Machine Learning  
✅ NLP Preprocessing  
✅ TF-IDF Vectorization  
✅ Logistic Regression  

Developed by Prince Kumar Saw
""")

st.sidebar.markdown("---")

st.sidebar.success("Model Accuracy: ~98%")

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; font-size:48px;'>
📰 AI-Powered Misinformation Detection
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px; color:lightgray;'>
Analyze news articles using Machine Learning and Natural Language Processing
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
news = st.text_area(
    "✍️ Enter News Article",
    height=250,
    placeholder="Paste your news article or headline here..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze News"):

    if news.strip() == "":
        st.warning("⚠️ Please enter some news text.")

    else:

        with st.spinner("Analyzing article with AI model..."):

            cleaned = clean_text(news)

            vector_input = vectorizer.transform([cleaned])

            prediction = model.predict(vector_input)

            probability = model.predict_proba(vector_input)

            real_probability = probability[0][0] * 100
            fake_probability = probability[0][1] * 100

            confidence = max(real_probability, fake_probability)

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        # RESULT DISPLAY
        if prediction[0] == 1:

            st.markdown(f"""
            <div class="result-box fake-news">
            🚨 FAKE NEWS DETECTED
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-box real-news">
            ✅ REAL NEWS DETECTED
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # CONFIDENCE CARD
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Prediction Confidence</h3>
            <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # PROBABILITIES
        st.subheader("📈 Probability Analysis")

        st.write(f"✅ Real News Probability: {real_probability:.2f}%")
        st.progress(int(real_probability))

        st.write(f"🚨 Fake News Probability: {fake_probability:.2f}%")
        st.progress(int(fake_probability))

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<p style='text-align:center; color:gray;'>
Built using Python • Machine Learning • NLP • Streamlit
</p>
""", unsafe_allow_html=True)