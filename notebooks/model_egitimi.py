# model_egitimi.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Türkçe stopword listesini indir ve yükle ---
nltk.download('stopwords')
turkce_stopwords = stopwords.words('turkish')

# --- 2. Veri setini oku ---
df = pd.read_csv("data/magaza_yorumlari_duygu_analizi.csv", encoding="utf-16")  
# encoding utf-16 çünkü dosyan UTF-16 olarak tespit edilmişti

# --- 3. Eksik verileri kontrol et ve temizle ---
print("Eksik veri sayısı:\n", df.isnull().sum())
df = df.dropna(subset=['Görüş'])  # Görüş sütununda eksik varsa sil

# --- 4. Veriyi X (özellik) ve y (etiket) olarak ayır ---
X = df['Görüş'].values  # yorumlar (metin)
y = df['Durum'].values   # etiket (Olumlu, Olumsuz, Tarafsız)

# --- 5. Veriyi eğitim ve test olarak ayır ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 6. TF-IDF vektörizer tanımla ve eğit ---
tfidf = TfidfVectorizer(max_features=5000, stop_words=turkce_stopwords)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# --- 7. Naive Bayes sınıflandırıcısını oluştur ve eğit ---
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Modeli ve TF-IDF vektörleştiricisini kaydet
import joblib
joblib.dump(model, "naivebayes_model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("Model ve TF-IDF vektörleştiricisi kaydedildi.")

# --- 8. Test verisi üzerinde tahmin yap ---
y_pred = model.predict(X_test_tfidf)

# --- 9. Sonuçları değerlendir ---
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred, zero_division=0))

# --- 10. Örnek tahmin yap (isteğe bağlı) ---
ornek_yorum = ["Ürün harika, çok beğendim!"]
ornek_tfidf = tfidf.transform(ornek_yorum)
tahmin = model.predict(ornek_tfidf)
print("\nÖrnek yorum tahmini:", tahmin[0])
