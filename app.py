"""
Aplikasi Web Deteksi URL Phishing
Menggunakan model Random Forest yang sudah dilatih (16 fitur).
"""

import re
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Deteksi URL Phishing",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    data = joblib.load('model_phishing_compressed.joblib')
    return data['model'], data['features'], data['model_name']









model, feature_cols, model_name = load_model()


# 16 fitur yang digunakan
SUSPICIOUS_WORDS = [
    'login', 'verify', 'secure', 'account', 'update', 'confirm',
    'banking', 'signin', 'password', 'paypal', 'ebay', 'amazon',
    'free', 'gift', 'winner', 'prize', 'urgent', 'alert', 'suspend'
]

BRAND_NAMES = [
    'paypal', 'google', 'facebook', 'apple', 'microsoft', 'amazon',
    'netflix', 'instagram', 'whatsapp', 'bank', 'bca', 'mandiri', 'bri'
]

SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']


def extract_features(url: str) -> dict:
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


#tampilan
st.title("🛡️ Deteksi URL Phishing")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Random Forest)** untuk mendeteksi
apakah sebuah URL berpotensi **Phishing** atau **Aman**.

**Cara pakai:** Masukkan URL di bawah, lalu klik tombol **Periksa**.
""")

st.divider()

url_input = st.text_input(
    "🔗 Masukkan URL yang ingin diperiksa:",
    placeholder="contoh: http://secure-login.paypal.verify-account.com"
)

if st.button("🔍 Periksa URL", type="primary", use_container_width=True):
    if not url_input.strip():
        st.warning("⚠️ Mohon masukkan URL terlebih dahulu.")
    else:
        features = extract_features(url_input)
        X_input = pd.DataFrame([features])[feature_cols]

        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]

        st.divider()

        if prediction == 1:
            st.error("# 🚨 PHISHING TERDETEKSI BROK WASPADA NIGGA!")
            st.markdown(f"""
            ### ⚠️ PERINGATAN!
            URL yang kamu masukkan **terindikasi kuat sebagai PHISHING**.
            
            **Tingkat Keyakinan Model:** `{probability[1] * 100:.2f}%`
            
            **JANGAN** memasukkan informasi pribadi, password, atau data
            perbankan di situs ini!
            """)
        else:
            st.success("# ✅ URL TERLIHAT AMAN SANTAI DULU SLUR")
            st.markdown(f"""
            ### 👍🏿 Kemungkinan Aman
            URL yang kamu masukkan **terlihat aman** menurut model kami.
            
            **Tingkat Keyakinan Model:** `{probability[0] * 100:.2f}%`
            
            *Catatan: Tetap waspada ya. Model ini tidak 100% sempurna.*
            """)

        with st.expander("🔬 Lihat Detail Analisis Fitur"):
            st.markdown("**Fitur yang diekstrak dari URL kamu:**")
            feat_df = pd.DataFrame([features]).T
            feat_df.columns = ['Nilai']
            st.dataframe(feat_df, use_container_width=True)

st.divider()
st.caption(f"Model yang digunakan: **{model_name}** | Proyek Machine Learning - Deteksi Phishing")