import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

nltk.download('stopwords')

model_path = 'random_forest_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

df = pd.read_csv("news_dataset.csv")

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = text.split()
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)
    return ''

df['text'] = df['text'].apply(preprocess_text)
X = df['text']
y = df['label'].map({'REAL': 0, 'FAKE': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    best_rf = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    print("Loaded model and vectorizer from disk")
else:
    tfidf_vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_tfidf, y_train)
    best_rf = grid_search.best_estimator_

    joblib.dump(best_rf, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print("Model and vectorizer trained and saved")

y_pred = best_rf.predict(tfidf_vectorizer.transform(X_test))
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

def predict_fake_news(text):
    processed_text = preprocess_text(text)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    prediction = best_rf.predict(vectorized_text)
    return "FAKE" if prediction[0] == 1 else "REAL"
