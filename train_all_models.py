import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, f1_score, confusion_matrix
from pythainlp.tokenize import word_tokenize
import numpy as np

# 1. Load Dataset
print("=" * 50)
print("Loading dataset...")
print("=" * 50)
try:
    df = pd.read_csv('dataset1.csv')
except FileNotFoundError:
    print("Error: dataset1.csv not found.")
    exit()

# Ensure required columns exist
required_cols = ['text', 'rating', 'category', 'sentiment', 'sarcasm']
for col in required_cols:
    if col not in df.columns:
        print(f"Error: Dataset must contain '{col}' column.")
        exit()

# Drop missing values
df = df.dropna(subset=required_cols)

print(f"Dataset size: {len(df)} samples")
print(f"\nColumn distributions:")
print(f"  Rating: {df['rating'].value_counts().sort_index().to_dict()}")
print(f"  Sentiment: {df['sentiment'].value_counts().to_dict()}")
print(f"  Category: {len(df['category'].unique())} categories")
print(f"  Sarcasm: {df['sarcasm'].value_counts().to_dict()}")

X = df['text']
y_rating = df['rating']
y_category = df['category']
y_sentiment = df['sentiment']
y_sarcasm = df['sarcasm']

# 2. Split Data
print("\n" + "=" * 50)
print("Splitting data (80% train, 20% test)...")
print("=" * 50)
# We need to keep indices to track back the text for Error Analysis
indices = np.arange(len(X))
X_train, X_test, y_rat_train, y_rat_test, y_cat_train, y_cat_test, y_sent_train, y_sent_test, y_sarc_train, y_sarc_test, idx_train, idx_test = train_test_split(
    X, y_rating, y_category, y_sentiment, y_sarcasm, indices, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 3. Define Tokenizer Function
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

# 4. Create TF-IDF Vectorizer
print("\n" + "=" * 50)
print("Creating TF-IDF vectorizer...")
print("=" * 50)
tfidf = TfidfVectorizer(tokenizer=thai_tokenizer, ngram_range=(1, 2), min_df=3, max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# =============================================
# 5. Train Rating Model (Regression)
# =============================================
print("\n" + "=" * 50)
print("Training RATING model (Ridge Regression)...")
print("=" * 50)
rating_model = Ridge(alpha=1.0, random_state=42)
rating_model.fit(X_train_tfidf, y_rat_train)

y_rat_pred = rating_model.predict(X_test_tfidf)
y_rat_pred_clipped = np.clip(y_rat_pred, 1, 5)

print(f"  MSE: {mean_squared_error(y_rat_test, y_rat_pred_clipped):.4f}")
print(f"  MAE: {mean_absolute_error(y_rat_test, y_rat_pred_clipped):.4f}")
print(f"  R2 Score: {r2_score(y_rat_test, y_rat_pred_clipped):.4f}")

# =============================================
# 6. Train Category Model (Classification)
# =============================================
print("\n" + "=" * 50)
print("Training CATEGORY model (Logistic Regression)...")
print("=" * 50)
category_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced')
category_model.fit(X_train_tfidf, y_cat_train)

y_cat_pred = category_model.predict(X_test_tfidf)
print(f"  Accuracy: {accuracy_score(y_cat_test, y_cat_pred):.4f}")
print(f"  Macro F1: {f1_score(y_cat_test, y_cat_pred, average='macro'):.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_cat_test, y_cat_pred)}")

# =============================================
# 7. Train Sentiment Model (Classification)
# =============================================
print("\n" + "=" * 50)
print("Training SENTIMENT model (Logistic Regression)...")
print("=" * 50)
sentiment_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced')
sentiment_model.fit(X_train_tfidf, y_sent_train)

y_sent_pred = sentiment_model.predict(X_test_tfidf)
print(f"  Accuracy: {accuracy_score(y_sent_test, y_sent_pred):.4f}")
print(f"  Macro F1: {f1_score(y_sent_test, y_sent_pred, average='macro'):.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_sent_test, y_sent_pred)}")


# --- Error Analysis (Sentiment) ---
print("\n[Error Analysis - Sentiment] Showing 10 Incorrect Predictions:")
print("-" * 60)
errors = []
for i, (true_label, pred_label) in enumerate(zip(y_sent_test, y_sent_pred)):
    if true_label != pred_label:
        original_text = X.iloc[idx_test[i]]
        errors.append({
            "text": original_text,
            "actual": true_label,
            "predicted": pred_label
        })

for i, err in enumerate(errors[:10]):
    print(f"{i+1}. Text: {err['text'][:80]}...")
    print(f"   Actual: {err['actual']} | Predicted: {err['predicted']}")
print("-" * 60)

# =============================================
# 8. Train Sarcasm Model (Binary Classification)
# =============================================
print("\n" + "=" * 50)
print("Training SARCASM model (Logistic Regression)...")
print("=" * 50)
sarcasm_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced')
sarcasm_model.fit(X_train_tfidf, y_sarc_train)

y_sarc_pred = sarcasm_model.predict(X_test_tfidf)
print(f"  Accuracy: {accuracy_score(y_sarc_test, y_sarc_pred):.4f}")
print(f"  Macro F1: {f1_score(y_sarc_test, y_sarc_pred, average='macro'):.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_sarc_test, y_sarc_pred)}")

# =============================================
# 9. Save All Models (using joblib as required)
# =============================================
print("\n" + "=" * 50)
print("Saving all models to .joblib...")
print("=" * 50)

models_to_save = {
    'tfidf_vectorizer.joblib': tfidf,
    'rating_model.joblib': rating_model,
    'category_model.joblib': category_model,
    'sentiment_model.joblib': sentiment_model,
    'sarcasm_model.joblib': sarcasm_model
}

for filename, model in models_to_save.items():
    joblib.dump(model, filename)
    print(f"  Saved: {filename}")

# Save Errors for Web App (Requirement: Error Example Page)
import json
with open('errors.json', 'w', encoding='utf-8') as f:
    json.dump(errors[:20], f, ensure_ascii=False, indent=2) # Save top 20 errors
print("  Saved: errors.json")

print("\n" + "=" * 50)
print("Training complete! All models saved successfully.")
print("=" * 50)

