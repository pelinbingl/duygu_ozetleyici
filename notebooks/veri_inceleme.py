import pandas as pd

# Veriyi oku
df = pd.read_csv("data/magaza_yorumlari_duygu_analizi.csv", encoding="utf-16")

# Kolon adlarını düzelt (garanti olsun diye)
df = df.rename(columns={"Görüş": "yorum", "Durum": "puan"})

# İlk 5 satırı göster
print("İlk 5 Satır:\n", df.head())

# Veri kümesinde kaç yorum var?
print("\nToplam yorum sayısı:", len(df))

# Puan dağılımını göster
print("\nPuan Dağılımı:")
print(df["puan"].value_counts().sort_index())

# Eksik veri var mı?
print("\nEksik veri kontrolü:")
print(df.isnull().sum())
# 1. Eksik yorumları kaldır
df = df.dropna(subset=["yorum"])

# 2. Duygu etiketlerini sayıya çevir
etiket_map = {"Olumsuz": 0, "Tarafsız": 1, "Olumlu": 2}
df["duygu"] = df["puan"].map(etiket_map)

# 3. Kontrol edelim
print("\nYeni etiketli veri:")
print(df[["yorum", "puan", "duygu"]].head())

# 4. Her sınıftan kaç tane var
print("\nSınıf dağılımı:")
print(df["duygu"].value_counts().sort_index())

