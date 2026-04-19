import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. CREATE MOCK DATA
# In real projects: df = pd.read_csv('flipkart_reviews.csv')
# ==========================================
data = {
    'Review': [
        "Absolutely wonderful! The phone works perfectly and battery life is great.",
        "Terrible product. The screen broke within 2 days of normal use.",
        "Waste of money. Customer service was also very rude to me.",
        "Good value for the price. I highly recommend this to everyone.",
        "Camera quality is extremely poor, very disappointed.",
        "Fast delivery and genuine product. Five stars!",
        "It hangs a lot and heats up quickly. Do not buy.",
        "Nice design, smooth performance. Totally loved it.",
        "The charger stopped working after a week. Bad quality.",
        "Best purchase ever, exactly as described on the website."
    ],
    # 1: Positive, 0: Negative
    'Sentiment': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1] 
}

df = pd.DataFrame(data)

print("Sample data:")
print(df.head())
print("-" * 50)

# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to the whole Review column
df['Cleaned_Review'] = df['Review'].apply(clean_text)

# Separate Features (X) and Target (y)
X = df['Cleaned_Review']
y = df['Sentiment']

# ==========================================
# 3. TEXT TO NUMERIC (TF-IDF VECTORIZATION)
# ==========================================
# Initialize TF-IDF (Term Frequency–Inverse Document Frequency)
# Remove stopwords (stop_words='english') such as 'the', 'is', 'in', etc.
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X_tfidf = tfidf.fit_transform(X)

# Split into Train / Test (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
# Use Logistic Regression because it performs well and fast for binary text classification
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 5. MODEL EVALUATION
# ==========================================
y_pred = model.predict(X_test)

print("\nACCURACY:")
print(round(accuracy_score(y_test, y_pred), 4))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))

# ==========================================
# 6. PREDICTING NEW REVIEWS
# ==========================================
new_reviews = [
    "I am very happy with this laptop, it runs very fast!",
    "Worst delivery experience, the box was totally crushed and missing items."
]

# Clean the new reviews
cleaned_new_reviews = [clean_text(review) for review in new_reviews]

# Transform using TF-IDF (Important: use .transform(), not .fit_transform())
new_reviews_tfidf = tfidf.transform(cleaned_new_reviews)

# Predict
predictions = model.predict(new_reviews_tfidf)

print("\nPREDICTION RESULTS FOR NEW REVIEWS:")
for review, sentiment in zip(new_reviews, predictions):
    label = "Positive 🟢" if sentiment == 1 else "Negative 🔴"
    print(f"- Review: '{review}' -> Sentiment: {label}")