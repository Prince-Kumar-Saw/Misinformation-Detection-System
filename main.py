import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords', quiet=True)

# Stemmer
ps = PorterStemmer()

# Stopwords
stop_words = set(stopwords.words('english'))

print("Loading datasets...")

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Labels
# 1 = FAKE
# 0 = REAL
fake["label"] = 1
true["label"] = 0

# Merge datasets
data = pd.concat([fake, true])

# Shuffle
data = data.sample(frac=1, random_state=42)

# Keep needed columns
data = data[["title", "text", "label"]]

# Combine text
data["content"] = data["title"] + " " + data["text"]

print("Preprocessing started...")

# Clean text function
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

# Apply preprocessing
data["content"] = data["content"].apply(clean_text)

print("Preprocessing completed")

# Inputs and outputs
X = data["content"]
y = data["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

print("TF-IDF completed")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)

print("Training model...")

# Train
model.fit(X_train, y_train)

print("Training completed")

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("models/model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully")