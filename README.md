# 🧠 Duyguya Duyarlı Otomatik Özetleme

Bu proje, Türkçe mağaza yorumlarını analiz ederek duygu (olumlu / olumsuz / tarafsız) sınıflandırması yapan basit bir makine öğrenimi modelini içerir. Ayrıca kullanıcıdan alınan yorumlar Flask tabanlı bir web arayüzü ile tahmin edilir.

## 🚀 Proje Özellikleri

- Türkçe yorum verisi üzerinden TF-IDF vektörleştirme
- Naive Bayes ile duygu sınıflandırması
- Eğitimli modelin `.joblib` ile kaydedilmesi ve yeniden kullanımı
- Flask ile kullanıcı arayüzü


## 🔬 Kullanılan Teknolojiler

- **Python**  
- **pandas**, **scikit-learn**, **joblib**  
- **Flask** (basit web arayüzü için)  
- **TF-IDF**: Metinleri sayısal vektörlere dönüştürmek için  
- **Naive Bayes**: Duygu sınıflandırması için temel makine öğrenmesi algoritması

---

## 🚀 Projeyi Çalıştırmak

1. Gerekli paketleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

2. Flask uygulamasını başlatın:
    ```bash
    python notebooks/app.py
    ```

3. Web tarayıcınızda şu adrese gidin:  
    👉 http://127.0.0.1:5000

---

## 📌 Örnek Kullanım

Kullanıcı arayüzüne bir yorum girilir:

> **"Ürün çok kaliteli, kargo hızlıydı."**

Model bu yorumu **pozitif** olarak sınıflandırır ✅

---

## 👩‍💻 Geliştirici

**Pelin Bingöl**  
🔗 [GitHub Profilim](https://github.com/pelinbingl)

---

## 🧠 Geliştirme Fikirleri

- Daha gelişmiş bir model: `BERT`, `Logistic Regression`, `Random Forest`
- Veri temizleme modülü (stopword, lemmatization, vs.)
- Otomatik özetleme modülü (TextRank, Pegasus)
- Çoklu dil desteği (Türkçe + İngilizce)

---

## 📝 Lisans

Bu proje açık kaynaklıdır. Her türlü katkıya açıktır.
