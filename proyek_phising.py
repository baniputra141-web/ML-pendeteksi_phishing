"""
Proyek Machine Learning: Deteksi URL Phishing
Dataset: Phishing Site URLs (5.49 juta baris)
Alur: Feature Engineering -> Perbandingan Model -> Evaluasi -> Simpan Model
"""

import os
import re
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("1. MEMUAT DATASET")
print("=" * 60)

df = pd.read_csv('./data/phishing_site_urls.csv')
print(f"Total baris: {len(df):,}")
print(f"Kolom: {df.columns.tolist()}")
print(f"\nDistribusi kelas:\n{df['Label'].value_counts()}")

# Bersihkan data
df = df.dropna().drop_duplicates()
print(f"\nSetelah cleaning: {len(df):,} baris")


# ============================================================
# 2. FEATURE ENGINEERING (Ekstraksi fitur dari URL)
# ============================================================
print("\n" + "=" * 60)
print("2. FEATURE ENGINEERING (VERSI UPGRADE)")
print("=" * 60)

# Daftar kata-kata yang sering muncul di URL phishing
SUSPICIOUS_WORDS = [
    'login', 'verify', 'secure', 'account', 'update', 'confirm',
    'banking', 'signin', 'password', 'paypal', 'ebay', 'amazon',
    'free', 'gift', 'winner', 'prize', 'urgent', 'alert', 'suspend'
]

# Daftar merek terkenal yang sering dipalsukan
BRAND_NAMES = [
    'paypal', 'google', 'facebook', 'apple', 'microsoft', 'amazon',
    'netflix', 'instagram', 'whatsapp', 'bank', 'bca', 'mandiri', 'bri'
]

# Daftar layanan pemendek URL
SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']


def extract_features(url: str) -> dict:
    """Ekstrak fitur struktural dan leksikal dari sebuah URL."""
    url_lower = str(url).lower()
    url_str = str(url)

    
    domain_match = re.search(r'https?://([^/]+)', url_lower)
    domain = domain_match.group(1) if domain_match else url_lower

    return {
        'url_length': len(url_str),
        'num_dots': url_str.count('.'),
        'num_hyphens': url_str.count('-'),
        'num_slashes': url_str.count('/'),
        'num_digits': sum(c.isdigit() for c in url_str),
        'num_special': len(re.findall(r'[?=&%_@]', url_str)),
        'has_ip': 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_str) else 0,
        'has_https': 1 if url_str.startswith('https') else 0,
        'has_at': 1 if '@' in url_str else 0,
        'num_subdomains': max(0, url_str.count('.') - 1),
        'digit_ratio': sum(c.isdigit() for c in url_str) / max(len(url_str), 1),

        'has_suspicious_words': 1 if any(w in url_lower for w in SUSPICIOUS_WORDS) else 0,
        'has_brand_name': 1 if any(b in domain for b in BRAND_NAMES) else 0,
        'domain_length': len(domain),
        'has_consecutive_digits': 1 if re.search(r'\d{3,}', url_str) else 0,
        'is_shortened': 1 if any(s in domain for s in SHORTENERS) else 0,
    }

print("Mengekstrak fitur dari setiap URL... (ini butuh waktu ~1-2 menit)")
features_df = df['URL'].apply(extract_features).apply(pd.Series)
df = pd.concat([df, features_df], axis=1)

# Encode label: 'good' -> 0, 'bad' -> 1
df['target'] = df['Label'].map({'good': 0, 'bad': 1})
print(f"Total fitur yang diekstrak: {len(features_df.columns)}")
print(f"Daftar fitur: {list(features_df.columns)}")
print(f"Contoh data:\n{df[['URL', 'url_length', 'has_suspicious_words', 'has_brand_name', 'target']].head()}")


# ============================================================
# 3. SPLIT DATA
# ============================================================
print("\n" + "=" * 60)
print("3. SPLIT DATA (80% Train, 20% Test)")
print("=" * 60)

feature_cols = list(features_df.columns)
X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")


# ============================================================
# 4. LATIH & BANDINGKAN MODEL
# ============================================================
print("\n" + "=" * 60)
print("4. MELATIH DAN MEMBANDINGKAN MODEL")
print("=" * 60)

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    ),
}

results = {}
for name, model in models.items():
    print(f"\n>>> Melatih {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
    }

    print(f"Accuracy : {results[name]['Accuracy']:.4f}")
    print(f"Precision: {results[name]['Precision']:.4f}")
    print(f"Recall   : {results[name]['Recall']:.4f}")
    print(f"F1-Score : {results[name]['F1-Score']:.4f}")


# ============================================================
# 5. VISUALISASI PERBANDINGAN
# ============================================================
print("\n" + "=" * 60)
print("5. VISUALISASI HASIL")
print("=" * 60)

# Tabel perbandingan
results_df = pd.DataFrame(results).T
print("\nTabel Perbandingan Model:")
print(results_df.round(4))

# Bar chart perbandingan
results_df.plot(kind='bar', figsize=(10, 6), colormap='viridis', edgecolor='black')
plt.title('Perbandingan Performa Model - Deteksi Phishing')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.ylim(0.8, 1.02)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('perbandingan_model.png', dpi=150)
plt.close()
print("Grafik perbandingan disimpan: perbandingan_model.png")

# Confusion Matrix untuk model terbaik
best_model_name = results_df['F1-Score'].idxmax()
print(f"\nModel terbaik berdasarkan F1-Score: {best_model_name}")

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=['Legitimate', 'Phishing'],
    cmap='Blues'
)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix disimpan: confusion_matrix.png")

print("\nClassification Report (Model Terbaik):")
print(classification_report(y_test, y_pred_best,
                            target_names=['Legitimate', 'Phishing']))


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE")
print("=" * 60)

if "Random Forest" in models:
    rf_model = models["Random Forest"]
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False)
    print("\nTop Fitur Paling Penting:")
    print(imp)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=imp.values, y=imp.index, hue=imp.index,
                palette='viridis', legend=False)
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("\nGrafik feature importance disimpan: feature_importance.png")


# ============================================================
# 7. SIMPAN MODEL TERBAIK
# ============================================================
print("\n" + "=" * 60)
print("7. MENYIMPAN MODEL")
print("=" * 60)

joblib.dump({
    'model': best_model,
    'model_name': best_model_name,
    'features': feature_cols
}, 'model_phishing.joblib')

print(f"Model terbaik ({best_model_name}) disimpan ke: model_phishing.joblib")
print("\n" + "=" * 60)
print("PROYEK SELESAI!")
print("=" * 60)