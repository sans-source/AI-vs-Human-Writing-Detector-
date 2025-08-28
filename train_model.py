import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (AI vs Human text)
texts = [
    "Artificial intelligence is transforming the world with amazing speed and efficiency.",  # AI
    "The sun was setting behind the hills, casting a golden glow over the quiet village.",   # Human
    "In this study, we propose a novel deep learning method for text classification.",       # AI
    "I went for a walk in the park yesterday and enjoyed the fresh air.",                   # Human
]

labels = [1, 0, 1, 0]  # 1 = AI, 0 = Human

# Step 1: Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Train model
model = LogisticRegression()
model.fit(X, labels)

# Step 3: Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")