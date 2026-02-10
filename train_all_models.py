import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
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
# Note: Naive Bayes requires non-negative values. TF-IDF is fine.
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

# =============================================
# 7. Train Sentiment Models (Comparison)
# =============================================
print("\n" + "=" * 50)
print("Training SENTIMENT models (Comparison)...")
print("=" * 50)

sentiment_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced'),
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(random_state=42, class_weight='balanced', dual=False), # LinearSVC is faster for text
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
}

best_f1 = 0
best_model_name = ""
trained_sentiment_models = {}

for name, model in sentiment_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_sent_train)
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_sent_test, y_pred)
    f1 = f1_score(y_sent_test, y_pred, average='macro')
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    
    trained_sentiment_models[name] = model
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

print(f"\n>> Best Sentiment Model: {best_model_name} (F1: {best_f1:.4f})")

# =============================================
# 8. Train Sarcasm Model (Binary Classification)
# =============================================
print("\n" + "=" * 50)
print("Training SARCASM model (Logistic Regression)...")
print("=" * 50)
# Keeping LogReg for Sarcasm as base, but could also compare if requested. For now, stick to original.
sarcasm_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced')
sarcasm_model.fit(X_train_tfidf, y_sarc_train)

y_sarc_pred = sarcasm_model.predict(X_test_tfidf)
print(f"  Accuracy: {accuracy_score(y_sarc_test, y_sarc_pred):.4f}")
print(f"  Macro F1: {f1_score(y_sarc_test, y_sarc_pred, average='macro'):.4f}")

# =============================================
# 9. Save All Models
# =============================================
print("\n" + "=" * 50)
print("Saving all models to .joblib...")
print("=" * 50)

# 1. Base Models
models_to_save = {
    'tfidf_vectorizer.joblib': tfidf,
    'rating_model.joblib': rating_model,
    'category_model.joblib': category_model,
    'sarcasm_model.joblib': sarcasm_model,
    # Save the BEST sentiment model as the default 'sentiment_model.joblib' for backward compatibility
    'sentiment_model.joblib': trained_sentiment_models[best_model_name]
}

# 2. Save Specific Sentiment Models for Comparison
models_to_save['sentiment_model_logreg.joblib'] = trained_sentiment_models['Logistic Regression']
models_to_save['sentiment_model_nb.joblib'] = trained_sentiment_models['Naive Bayes']
models_to_save['sentiment_model_svm.joblib'] = trained_sentiment_models['SVM']
models_to_save['sentiment_model_rf.joblib'] = trained_sentiment_models['Random Forest']

for filename, model in models_to_save.items():
    joblib.dump(model, filename)
    print(f"  Saved: {filename}")

# Save Errors (using the Best Model's predictions for the error page)
best_model = trained_sentiment_models[best_model_name]
y_sent_pred_best = best_model.predict(X_test_tfidf)

errors = []
for i, (true_label, pred_label) in enumerate(zip(y_sent_test, y_sent_pred_best)):
    if true_label != pred_label:
        original_text = X.iloc[idx_test[i]]
        errors.append({
            "text": original_text,
            "actual": true_label,
            "predicted": pred_label
        })

import json
with open('errors.json', 'w', encoding='utf-8') as f:
    json.dump(errors[:20], f, ensure_ascii=False, indent=2) 
print("  Saved: errors.json (Generated from Best Model)")

print("\n" + "=" * 50)
print("Training complete! All comparison models saved.")
print("=" * 50)

