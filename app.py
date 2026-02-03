import os
import sys
import joblib
import time
import numpy as np
from flask import Flask, request, jsonify, render_template
from pythainlp.tokenize import word_tokenize

# Define tokenizer (must match the one used during training)
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

# Hack to allow joblib/pickle to find 'thai_tokenizer' if it expects it in __main__
import __main__
__main__.thai_tokenizer = thai_tokenizer

app = Flask(__name__)

# Load All Models
tfidf = None
rating_model = None
category_model = None
sentiment_model = None
sarcasm_model = None

try:
    print("Loading models...")
    
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    print("  TF-IDF vectorizer loaded.")
    
    rating_model = joblib.load('rating_model.joblib')
    print("  Rating model loaded.")
    
    category_model = joblib.load('category_model.joblib')
    print("  Category model loaded.")
    
    sentiment_model = joblib.load('sentiment_model.joblib')
    print("  Sentiment model loaded.")
    
    sarcasm_model = joblib.load('sarcasm_model.joblib')
    print("  Sarcasm model loaded.")
    
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run 'python train_all_models.py' first.")

import pandas as pd

# Load Dataset
try:
    df = pd.read_csv('dataset1.csv')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame()

# ... (Previous imports exist, but allow this block to fit naturally)

@app.route('/')
def index():
    # Get unique categories
    if not df.empty:
        categories = df['category'].unique().tolist()
    else:
        categories = []
    return render_template('home.html', categories=categories)

@app.route('/directory')
def directory():
    if df.empty:
        return "Dataset not loaded", 500
    # Pass all items to the directory page
    items = df.to_dict('records')
    return render_template('directory.html', items=items)

@app.route('/category/<path:category_name>')
def category_page(category_name):
    if df.empty:
        return "Dataset not loaded", 500
    
    # Filter by category
    items = df[df['category'] == category_name].to_dict('records')
    # Limit to 7 items as requested
    items = items[:7]
    return render_template('restaurants.html', category=category_name, items=items)

@app.route('/review/<review_id>')
def review_page(review_id):
    if df.empty:
        return "Dataset not loaded", 500
    
    # Find review by ID
    # Note: dataset1.csv uses 'review_id'. Ensure string matching.
    item = df[df['review_id'] == review_id].to_dict('records')
    
    if not item:
        return "Review not found", 404
    
    # Generate a stable integer from review_id for the image lock
    # detailed enough to vary, stable enough to persist
    image_lock = sum(ord(c) for c in review_id)
        
    return render_template('review.html', item=item[0], image_lock=image_lock)

# Keep existing health check below

# Smart Search Logic
KEYWORD_MAP = {
    'อาหารอีสาน': ['ลาบ', 'ส้มตำ', 'ไก่ย่าง', 'น้ำตก', 'แจ่วฮ้อน', 'ตับหวาน', 'ปลาร้า', 'ยำ', 'แซ่บ','อาหารอีสาน','อีสาน','อีสาน'],
    'อาหารญี่ปุ่น': ['ซูชิ', 'ซาชิมิ', 'ราเมง', 'ข้าวหน้า', 'เทมปุระ', 'อูด้ง', 'ปลาดิบ', 'ญี่ปุ่น', 'แซลมอน','ญี่ปุ่น','ญี่ปุ่น'],
    'ก๋วยเตี๋ยว': ['เส้น', 'บะหมี่', 'เกี๊ยว', 'เย็นตาโฟ', 'เนื้อเปื่อย', 'ไก่ตุ๋น', 'ลูกชิ้น', 'ต้มยำ','ก๋วยเตี๋ยว',],
    'คาเฟ่': ['กาแฟ', 'โกโก้', 'ลาเต้', 'มัทฉะ', 'เค้ก', 'โทสต์', 'ของหวาน', 'เบเกอรี่', 'ขนมปัง', 'ชา','คาเฟ่'],
    'ปิ้งย่าง': ['หมูกระทะ', 'ปิ้งย่าง', 'เนื้อย่าง', 'ยากินิกุ', 'บุฟเฟ่ต์', 'ปิ้ง','ย่าง'],
    'ชาบู': ['สุกี้', 'ชาบู', 'หม่าล่า', 'หม้อไฟ', 'จิ้มจุ่ม','ชาบู'],
    'อาหารจีน': ['ติ่มซำ', 'ซาลาเปา', 'เป็ดย่าง', 'ฮะเก๋า', 'จีน', 'เกี๊ยวซ่า','อาหารจีน','จีน'],
    'อาหารใต้': ['แกงไตปลา', 'คั่วกลิ้ง', 'ผัดสะตอ', 'ใบเหลียง', 'ใต้', 'ขมิ้น'],
    'อาหารตามสั่ง': ['กะเพรา', 'ข้าวผัด', 'คะน้า', 'หมูทอด', 'กระเทียม', 'ผัด', 'ทอด','อาหารตามสั่ง','ตามสั่ง'],
    'สเต๊ก': ['สเต๊ก', 'เนื้อ', 'พอร์คชอป', 'สลัด', 'มันบด', 'ผักโขม'],
}

# Static Image URLs for consistent quality
IMAGE_KEYWORDS = {
    'อาหารอีสาน': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQsUp5uCAJhJt604BdOS0bDmJdt4pTHDHeY91lmbHIEeovwzw0hCKBTtXmiaTDDj4GTof7-ZyCNslUOsxbIDnqPMP3CRNOygrDbJS7s5Bc&s=10', # Som Tum
    'อาหารญี่ปุ่น': 'https://images.unsplash.com/photo-1579871494447-9811cf80d66c?auto=format&fit=crop&w=800&q=80', # Sushi
    'ก๋วยเตี๋ยว': 'https://images.unsplash.com/photo-1555126634-323283e090fa?auto=format&fit=crop&w=800&q=80', # Noodle Soup
    'คาเฟ่': 'https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&w=800&q=80', # Coffee
    'ปิ้งย่าง': 'https://images.unsplash.com/photo-1555939594-58d7cb561ad1?auto=format&fit=crop&w=800&q=80', # BBQ/Grill
    'ชาบู': 'https://images.unsplash.com/photo-1541544744-597830950e35?auto=format&fit=crop&w=800&q=80', # Hotpot (closest high quality)
    'อาหารจีน': 'https://cdn.ready-market.com.tw/21cd62de/Templates/pic/Chinese-Food-new.jpg?v=824c575d', # Dimsum
    'อาหารใต้': 'https://api2.krua.co/wp-content/uploads/2020/08/ImageBanner_1140x507-01-1.jpg', # Curry
    'อาหารตามสั่ง': 'https://patoisfdimage4-fcbugqebgmbma7he.z01.patois.com/patois/image/2023/10/19/PATOIS_2023-10-19_16_49_48_04f7dd8d-476d-4c12-bec3-d8b766a67151.jpg', # Basil Rice
    'สเต๊ก': 'https://images.unsplash.com/photo-1600891964092-4316c288032e?auto=format&fit=crop&w=800&q=80', # Steak
    'เครื่องดื่ม': 'https://images.unsplash.com/photo-1513558161293-cdaf765ed2fd?auto=format&fit=crop&w=800&q=80', # Drink
    'เบเกอรี่': 'https://images.unsplash.com/photo-1578985545062-69928b1d9587?auto=format&fit=crop&w=800&q=80', # Cake
}

# Individual Shop Images (Add shop_id: 'image_url' here)
SHOP_IMAGES = {
    # '7bb2c0f7-87aa-47b6-9c67-aac5102c1191': 'https://example.com/my-shop-image.jpg',
}

def get_shop_image(shop_id, category):
    # 1. Check if there's an individual shop image
    if shop_id in SHOP_IMAGES:
        return SHOP_IMAGES[shop_id]
    
    # 2. Fallback to category image
    return get_image_keyword(category)

def get_image_keyword(category):
    # Check for direct match
    for key, val in IMAGE_KEYWORDS.items():
        if key in category:
            return val
    # Additional checks
    if 'เค้ก' in category: return IMAGE_KEYWORDS['เบเกอรี่']
    if 'กาแฟ' in category: return IMAGE_KEYWORDS['คาเฟ่']
    
    # Default fallback
    return 'https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=800&q=80'

# Make it available to Jinja templates
app.jinja_env.globals.update(
    get_image_keyword=get_image_keyword,
    get_shop_image=get_shop_image
)

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return render_template('home.html', categories=df['category'].unique().tolist() if not df.empty else [])

    # Simple logic: Check keywords
    found_category = None
    
    # 1. Exact match with category
    categories = df['category'].unique().tolist() if not df.empty else []
    if query in categories:
        return category_page(query)

    # 2. Keyword match
    for cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in query:
                found_category = cat
                break
        if found_category:
            break
    
    if found_category:
        return category_page(found_category)
    
    # If no match found, just stay on home but maybe show an alert (optional)
    # For now, let's redirect to home
    return render_template('home.html', categories=categories, error="ไม่พบหมวดหมู่ที่ค้นหา ลองคำอื่นดูนะครับ")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Service is healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not tfidf or not rating_model or not category_model or not sentiment_model or not sarcasm_model:
        return jsonify({'error': 'Models not loaded'}), 500

    data = request.json
    
    # Support Headline + Body (Requirement Step 6) -> Removed Headline (User Request)
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided (headline or body required)'}), 400

    start_time = time.time()
    
    try:
        # Transform text using TF-IDF
        X = tfidf.transform([text])
        
        # =============================================
        # =============================================
        # Hybrid Logic: Keyword Matching overrides
        # =============================================
        neg_keywords = ['ไม่อร่อย', 'แย่มาก', 'ผิดหวัง', 'ไม่ประทับใจ', 'พูดจาไม่ดี', 'รอนานมาก', 'แย่', 'ห่วย']
        pos_keywords = ['อร่อยมาก', 'ดีมาก', 'ประทับใจ', 'สุดยอด', 'เยี่ยม', 'แนะนำเลย']
        neu_keywords = ['เฉยๆ', 'ธรรมดา', 'พอกินได้', 'ทั่วๆไป', 'ก็โอเค', 'ไม่ได้แย่', 'กลางๆ']
        
        # Check for strong keywords with Negation Handling
        is_strong_neg = False
        for kw in neg_keywords:
            if kw in text:
                # Fix: "ไม่ได้แย่" (Not bad) should not trigger "แย่" (Bad)
                if kw == 'แย่' and 'ไม่ได้แย่' in text:
                    continue
                is_strong_neg = True
                break
                
        is_strong_pos = False
        for kw in pos_keywords:
            if kw in text:
                # Fix: "ไม่ประทับใจ" (Not impressed) should not trigger "ประทับใจ" (Impressed)
                if kw == 'ประทับใจ' and 'ไม่ประทับใจ' in text:
                    continue
                # Fix: "ไม่อร่อย" (Not delicious) should not trigger "อร่อย"
                if 'อร่อย' in kw and 'ไม่อร่อย' in text:
                    continue
                is_strong_pos = True
                break
                
        is_strong_neu = any(kw in text for kw in neu_keywords)
        
        # Predict Rating (1-5 continuous)
        raw_rating = rating_model.predict(X)[0]

        # Determine sentiment label first to help adjust rating
        sentiment_pred = sentiment_model.predict(X)[0]
        sentiment_confidence = None
        if hasattr(sentiment_model, 'predict_proba'):
            probs = sentiment_model.predict_proba(X)[0]
            sentiment_confidence = max(probs)

        # Logic: Override/Boost rating based on keywords (Stronger than ML)
        # Fix: If both Strong Positive AND Strong Negative exist -> It's likely "Mixed" or specific complaint.
        # Don't force Negative if there are also Strong Positive words.
        if is_strong_neg and not is_strong_pos:
            if raw_rating > 2.5: raw_rating = 1.5 # Pull down hard
            sentiment_pred = 'Negative'
            sentiment_confidence = 0.95 
            
        elif is_strong_pos and not is_strong_neg:
            if raw_rating < 4.5: raw_rating = 4.5 # Pull up hard (v5.1 fix)
            sentiment_pred = 'Positive'
            if sentiment_confidence < 0.8: sentiment_confidence = 0.9
            
        elif is_strong_pos and is_strong_neg:
             # Conflict! (e.g. "Food is good but wait is long")
             # Let the ML model decide, but maybe cap the rating?
             # Or force Neutral? Let's trust the probabilities but ensure rating isn't extreme 1 or 5.
             if raw_rating > 4.0: raw_rating = 4.0
             if raw_rating < 2.0: raw_rating = 2.0
             # Don't override sentiment_pred, let ML handle the nuance.
             pass 
            
            
        elif is_strong_neu:
            # Force towards 3.0
            raw_rating = 3.0
            sentiment_pred = 'Neutral'
            sentiment_confidence = 0.90
            
        # Logic: ML-based boosting (If no strong keywords conflicted)
        elif sentiment_pred == 'Positive' and raw_rating < 4.0:
            raw_rating = (raw_rating + 4.5) / 2
        elif sentiment_pred == 'Negative' and raw_rating > 2.0:
            raw_rating = (raw_rating + 1.5) / 2

        # =============================================
        # Get Full Probabilities for UI (Synced with Logic)
        # =============================================
        sentiment_probs = {}
        
        # If we forced the sentiment via keywords, we must fake the probs to match
        
        # 1. Mixed Case (Pos + Neg): Don't fake it, let model speak (or force Neutral if you want)
        if is_strong_pos and is_strong_neg:
             # Use Model Probs (Fall through to else) or maybe average them? 
             # Let's fall through to 'else' to use real model probs.
             pass 
             
        # 2. Pure Strong Negative
        elif is_strong_neg:
            sentiment_probs = {'Positive': 0.05, 'Neutral': 0.0, 'Negative': 0.95}
            
        # 3. Pure Strong Positive
        elif is_strong_pos:
            sentiment_probs = {'Positive': 0.90, 'Neutral': 0.05, 'Negative': 0.05}
            
        # 4. Pure Strong Neutral
        elif is_strong_neu:
            sentiment_probs = {'Positive': 0.05, 'Neutral': 0.90, 'Negative': 0.05}
            
        if not sentiment_probs: # If pass above or no override
            # No strong override -> Use Model Probs
            if hasattr(sentiment_model, 'predict_proba'):
                classes = sentiment_model.classes_
                probs = sentiment_model.predict_proba(X)[0]
                for c, p in zip(classes, probs):
                    sentiment_probs[c] = float(p)
            else:
                # Fallback purely based on prediction
                sentiment_probs = {
                    'Positive': 1.0 if sentiment_pred == 'Positive' else 0.0,
                    'Neutral': 1.0 if sentiment_pred == 'Neutral' else 0.0,
                    'Negative': 1.0 if sentiment_pred == 'Negative' else 0.0,
                }

        # =============================================
        # GLOBAL SAFETY CHECK (Final Enforcer)
        # =============================================
        # User Feedback: "Positive" must be >= 4.0
        if sentiment_pred == 'Positive' and raw_rating < 4.0:
            raw_rating = 4.0
        
        # User Feedback: "Negative" must be <= 2.5
        if sentiment_pred == 'Negative' and raw_rating > 2.5:
            raw_rating = 2.5
        
        print(f"DEBUG: Text='{text}'") 
        print(f"DEBUG: Raw Rating={raw_rating}, Sentiment={sentiment_pred}, Probs={probs}")

        star_score = np.clip(raw_rating, 1, 5)
        star_score = round(star_score * 2) / 2  # Round to 0.5
        
        # =============================================
        # Predict Sarcasm
        # =============================================
        sarcasm_pred = sarcasm_model.predict(X)[0]
        sarcasm_confidence = None
        if hasattr(sarcasm_model, 'predict_proba'):
            sarc_probs = sarcasm_model.predict_proba(X)[0]
            sarcasm_confidence = max(sarc_probs)
        
        # Logic: Anti-Sarcasm Rules
        # 1. If explicit sarcasm marker "(ประชด)" exists -> Force True
        if "(ประชด)" in text:
            sarcasm_pred = True
            sarcasm_confidence = 1.0
        # 2. If Strong Positive Keywords are present -> Force False (Assume genuine praise)
        elif is_strong_pos:
            sarcasm_pred = False
        # 3. If Rating is high (>= 3.5), it's likely NOT sarcasm.
        # Only allow sarcasm if confidence is extremely high (> 95%)
        elif star_score >= 3.5 and sarcasm_pred and sarcasm_confidence < 0.95:
             sarcasm_pred = False
        
        # =============================================
        # Predict Category
        # =============================================
        category_pred = category_model.predict(X)[0]
        category_confidence = None
        if hasattr(category_model, 'predict_proba'):
            cat_probs = category_model.predict_proba(X)[0]
            category_confidence = max(cat_probs)
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return jsonify({
            # Rating
            'star_score': float(star_score),
            'raw_rating': float(raw_rating),
            
            # Sentiment
            'label': sentiment_pred,
            'sentiment': sentiment_pred,
            'sentiment_confidence': float(sentiment_confidence) if sentiment_confidence else None,
            'sentiment_probs': sentiment_probs,
            
            # Category
            'category': category_pred,
            'category_confidence': float(category_confidence) if category_confidence else None,
            
            # Sarcasm
            'sarcasm': bool(sarcasm_pred),
            'sarcasm_confidence': float(sarcasm_confidence) if sarcasm_confidence else None,
            
            # Meta
            'latency_ms': latency_ms,
            'version': 'v5.2 (Error Page + Highlights)',
            
            # Explainability (Important Words)
            'important_words': get_important_words(text, sentiment_pred)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/errors')
def show_errors():
    import json
    try:
        with open('errors.json', 'r', encoding='utf-8') as f:
            errors = json.load(f)
    except:
        errors = []
    return render_template('errors.html', errors=errors)

def get_important_words(text, label):
    """
    Find words that contributed most to the prediction.
    """
    try:
        tokens = thai_tokenizer(text)
        if not hasattr(sentiment_model, 'coef_'):
            return []
            
        feature_names = tfidf.get_feature_names_out()
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        # classes_ are usually ['Negative', 'Neutral', 'Positive']
        target_idx = np.where(sentiment_model.classes_ == label)[0][0]
        coefficients = sentiment_model.coef_[target_idx]
        
        # Define noise to filter out
        noise_words = set(['ๆ', 'นะ', 'คะ', 'ครับ', 'จ้ะ', 'เลย', 'น่ะ', 'นี้', 'มัน', 'ก็', 'ที่', '.', '/', ',', '(', ')'])
        
        important = []
        for token in tokens:
            # Filter noise
            if token in noise_words or token.isnumeric():
                continue
                
            if token in feature_map:
                idx = feature_map[token]
                score = coefficients[idx]
                if score > 0.5: # Threshold for "Important"
                    important.append({'word': token, 'score': float(score)})
        
        # Sort by impact
        important.sort(key=lambda x: x['score'], reverse=True)
        # Deduplicate by word
        seen = set()
        unique_important = []
        for item in important:
            if item['word'] not in seen:
                unique_important.append(item)
                seen.add(item['word'])
                
        return unique_important[:5] # Top 5
    except Exception as e:
        print(f"Explainability Error: {e}")
        return []

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'version': 'v5.2',
        'tokenizer': 'pythainlp (newmm)',
        'models': {
            'rating': 'Ridge Regression (1-5 stars)',
            'sentiment': 'Logistic Regression (Positive/Neutral/Negative)',
            'category': 'Logistic Regression (11 categories)',
            'sarcasm': 'Logistic Regression (True/False)'
        },
        'vectorizer': 'TF-IDF (shared joblib)'
    })

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, port=5000)

