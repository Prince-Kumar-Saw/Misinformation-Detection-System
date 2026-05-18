import streamlit as st
import pickle
import re
import nltk
import requests

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------------------
# DOWNLOAD NLTK
# -------------------------------
nltk.download("stopwords", quiet=True)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -------------------------------
# NEWS API KEY
# -------------------------------
API_KEY = "258a43483b6f47769a5592b40bffc13c"

# -------------------------------
# LOAD MODEL & VECTORIZER
# -------------------------------
@st.cache_resource
def load_models():

    with open("models/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("models/vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


model, vectorizer = load_models()

# -------------------------------
# NLP SETUP
# -------------------------------
stemmer = PorterStemmer()

stop_words = set(stopwords.words("english"))

# -------------------------------
# TRUSTED SOURCES
# -------------------------------
trusted_sources = [
    "BBC News",
    "BBC",
    "CNN",
    "NPR",
    "Reuters",
    "NBC News",
    "CBS News",
    "The New York Times",
    "The Washington Post",
    "Associated Press",
    "AP News",
    "The Guardian",
    "GSMArena.com",
    "Al Jazeera English",
    "ABC News",
    "Fox News",
    "KABC-TV",
    "CNBC",
    "The Hindu",
    "Hindustan Times",
    "Times of India",
    "NDTV",
    "India Today"
]

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_news(news_text):

    cleaned_text = clean_text(news_text)

    vector_input = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector_input)[0]

    probabilities = model.predict_proba(vector_input)[0]

    real_probability = probabilities[0] * 100

    fake_probability = probabilities[1] * 100

    confidence = max(real_probability, fake_probability)

    return (
        prediction,
        real_probability,
        fake_probability,
        confidence
    )

# -------------------------------
# FETCH LIVE NEWS
# -------------------------------
def fetch_live_news():

    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"country=us&"
        f"pageSize=8&"
        f"apiKey={API_KEY}"
    )

    try:

        response = requests.get(url, timeout=10)

        data = response.json()

        if data.get("status") != "ok":

            st.error(
                data.get(
                    "message",
                    "Unable to fetch news"
                )
            )

            return []

        articles = []

        for article in data.get("articles", []):

            title = article.get("title") or ""

            description = article.get("description") or ""

            content = article.get("content") or ""

            source = article.get("source", {}).get("name", "")

            full_text = f"{title} {description} {content}".strip()

            if full_text:

                articles.append({
                    "source": source,
                    "title": title,
                    "content": full_text
                })

        return articles

    except Exception as e:

        st.error(f"Error fetching news: {e}")

        return []

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>

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

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
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

.news-card {
    background-color: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 15px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🧠 About Project")

st.sidebar.info("""
This system detects whether news is REAL or FAKE using:

✅ NLP Preprocessing  
✅ TF-IDF Vectorization  
✅ Logistic Regression  
✅ Machine Learning  
✅ Live News Analysis
""")

st.sidebar.markdown("---")

st.sidebar.success("Model Accuracy: ~94%")

# -------------------------------
# TITLE
# -------------------------------
st.markdown("""
<h1 style='text-align:center; font-size:48px;'>
📰 AI-Powered Misinformation Detection
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px; color:lightgray;'>
Analyze news articles using Machine Learning and NLP
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# USER INPUT
# -------------------------------
news_input = st.text_area(
    "✍️ Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)

# -------------------------------
# ANALYZE USER NEWS
# -------------------------------
if st.button("🔍 Analyze News"):

    if news_input.strip() == "":

        st.warning("⚠️ Please enter some text.")

    else:

        with st.spinner("Analyzing article..."):

            (
                prediction,
                real_probability,
                fake_probability,
                confidence
            ) = predict_news(news_input)

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.markdown("""
            <div class="result-box fake-news">
            🚨 FAKE NEWS DETECTED
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-box real-news">
            ✅ REAL NEWS DETECTED
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Prediction Confidence</h3>
            <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📈 Probability Analysis")

        st.write(f"✅ Real News Probability: {real_probability:.2f}%")

        st.progress(int(real_probability))

        st.write(f"🚨 Fake News Probability: {fake_probability:.2f}%")

        st.progress(int(fake_probability))

# -------------------------------
# LIVE NEWS SECTION
# -------------------------------
st.markdown("---")

st.subheader("📰 Analyze Live Current News")

st.info(
    "Live news prediction is an ML-based credibility estimate. "
    "Short headlines may be less accurate than full-length articles."
)

if st.button("📡 Fetch Latest News"):

    with st.spinner("Fetching latest headlines..."):

        live_news = fetch_live_news()

    if len(live_news) == 0:

        st.warning("⚠️ No news articles found.")

    else:

        for idx, article in enumerate(live_news):

            source = article["source"]

            title = article["title"]

            content = article["content"]

            (
                prediction,
                real_probability,
                fake_probability,
                confidence
            ) = predict_news(content)

            # -------------------------------
            # SOURCE-AWARE CALIBRATION
            # -------------------------------
            if source in trusted_sources:

                real_probability += 35

                if real_probability > 98:
                    real_probability = 98

                fake_probability = 100 - real_probability

            confidence = max(
                real_probability,
                fake_probability
            )

            st.markdown("---")

            st.markdown(
                '<div class="news-card">',
                unsafe_allow_html=True
            )

            st.write(f"### 📰 News {idx + 1}")

            st.caption(f"Source: {source}")

            if title:
                st.write(f"**Title:** {title}")

            st.write(content)

            st.write(
                f"✅ Real News Probability: "
                f"{real_probability:.2f}%"
            )

            st.progress(int(real_probability))

            st.write(
                f"🚨 Fake News Probability: "
                f"{fake_probability:.2f}%"
            )

            st.progress(int(fake_probability))

            # -------------------------------
            # FINAL RESULT
            # -------------------------------
            if real_probability >= fake_probability:

                st.success(
                    f"✅ Likely Reliable News "
                    f"({confidence:.2f}%)"
                )

                st.info(
                    "This article appears credible based on "
                    "machine learning analysis and source evaluation."
                )

            else:

                st.warning(
                    f"⚠️ Potentially Suspicious News "
                    f"({confidence:.2f}%)"
                )

                st.info(
                    "This article contains linguistic patterns "
                    "commonly associated with misinformation."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")

st.markdown("""
<p style='text-align:center; color:gray;'>
Built using Python • NLP • Machine Learning • Streamlit • NewsAPI
</p>
""", unsafe_allow_html=True)