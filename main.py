import os
import re
import pickle
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# Download NLTK stopwords
# -------------------------------
nltk.download("stopwords", quiet=True)

# -------------------------------
# Create required folders
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# -------------------------------
# Initialize stemmer and stopwords
# -------------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# -------------------------------
# Text cleaning function
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
# Load datasets
# -------------------------------
print("Loading datasets...")

fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")

# Labels
# 1 = FAKE
# 0 = REAL
fake_news["label"] = 1
real_news["label"] = 0

# Merge datasets
data = pd.concat([fake_news, real_news], axis=0)

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep only required columns
data = data[["title", "text", "label"]]

# Handle missing values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Combine title and article text
data["content"] = data["title"] + " " + data["text"]

print("Dataset loaded successfully")
print("Total records:", len(data))
print("Fake news samples:", data[data["label"] == 1].shape[0])
print("Real news samples:", data[data["label"] == 0].shape[0])


# -------------------------------
# Preprocessing
# -------------------------------
print("\nPreprocessing started...")

data["clean_content"] = data["content"].apply(clean_text)

print("Preprocessing completed")


# -------------------------------
# Input and output
# -------------------------------
X = data["clean_content"]
y = data["label"]


# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# TF-IDF Vectorization
# -------------------------------
print("\nApplying TF-IDF vectorization...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed")


# -------------------------------
# Model training
# -------------------------------
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

model.fit(X_train_tfidf, y_train)

print("Training completed")


# -------------------------------
# Model evaluation
# -------------------------------
print("\nEvaluating model...")

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    target_names=["REAL", "FAKE"]
)
conf_matrix = confusion_matrix(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(conf_matrix)


# -------------------------------
# Save model and vectorizer
# -------------------------------
with open("models/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("models/vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel saved at: models/model.pkl")
print("Vectorizer saved at: models/vectorizer.pkl")


# -------------------------------
# Save evaluation report
# -------------------------------
with open("reports/evaluation_report.txt", "w") as file:
    file.write("AI-Powered Misinformation Detection System\n")
    file.write("-----------------------------------------\n\n")
    file.write(f"Accuracy: {round(accuracy * 100, 2)}%\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(conf_matrix))

print("Evaluation report saved at: reports/evaluation_report.txt")

print("\nTraining pipeline completed successfully!")