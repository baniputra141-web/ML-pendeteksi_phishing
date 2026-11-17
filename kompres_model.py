import joblib
import os

# 1. Muat model asli yang berukuran 135 MB
print("Memuat model asli...")
model_data = joblib.load('model_phishing.joblib')

# 2. Simpan ulang dengan kompresi zlib level 6 (keseimbangan kecepatan & ukuran)
print("Mengompres model...")
joblib.dump(
    model_data, 
    'model_phishing_compressed.joblib', 
    compress=('zlib', 6)  # <-- Kunci utamanya ada di sini
)

# 3. Cek perbandingan ukuran
ukuran_asli = os.path.getsize('model_phishing.joblib') / (1024 * 1024)
ukuran_baru = os.path.getsize('model_phishing_compressed.joblib') / (1024 * 1024)

print(f"Ukuran asli   : {ukuran_asli:.2f} MB")
print(f"Ukuran kompres: {ukuran_baru:.2f} MB")
print("Selesai! File model_phishing_compressed.joblib siap diunggah.")