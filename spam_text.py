import pandas as pd
import numpy as np
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Clean dataset
data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels
data['label'] = data['label'].map({'ham':0,'spam':1})

# Text preprocessing
def preprocess(text):

    text = text.lower()

    text = ''.join([char for char in text if char not in string.punctuation])

    words = text.split()

    words = [word for word in words if word not in stopwords.words('english')]

    return " ".join(words)

data['clean_message'] = data['message'].apply(preprocess)

# Feature extraction
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(data['clean_message'])

y = data['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

def predict_spam(message):

    message = preprocess(message)

    vector = tfidf.transform([message])

    prediction = model.predict(vector)

    if prediction == 1:
        return "Spam"
    else:
        return "Not Spam"


print("\nTest Message Result:")
print(predict_spam("Congratulations! You won a free lottery"))


nltk.download('stopwords')
