import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Datasets loaded successfully")

fake["label"] = 1
true["label"] = 0

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42)

print("Datasets merged & shuffled")

data = data[["title", "text", "label"]]
data["content"] = data["title"] + " " + data["text"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

print("Preprocessing...")

data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression (DEPLOYMENT MODEL)
log_model = LogisticRegression(max_iter=1000)

# SGD (for comparison only)
sgd_model = SGDClassifier(
    loss='hinge',
    penalty=None,
    learning_rate='optimal',
    max_iter=1000,
    random_state=42
)

print("Training Logistic Regression...")
log_model.fit(X_train, y_train)

print("Training SGD...")
sgd_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
sgd_pred = sgd_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("SGD Accuracy:", accuracy_score(y_test, sgd_pred))

# SAVE ONLY BEST MODEL FOR APP
pickle.dump(log_model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model saved successfully")