import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Spam Email Detection System")

# Load dataset (FIXED LINE)
df = pd.read_csv("spam.csv", sep='\t', header=None)

# Add column names
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham':0, 'spam':1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2
)

# Convert text → numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
msg = ["Congratulations! You won a free prize"]
msg_vec = vectorizer.transform(msg)
print("Prediction (1=Spam, 0=Not Spam):", model.predict(msg_vec)[0])