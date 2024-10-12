import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Veri setini oku
file_path = 'data/House Price Prediction Dataset.csv'  # Veri seti yolunuz
df = pd.read_csv(file_path)

# Verinin ilk satırlarını görüntüle
print(df.head())

# Veriyi sayısal değerlere dönüştür
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')

# NaN değerlerini sil
df.dropna(inplace=True)

# Fiyatın mod değeri
mod_value = df['Price'].mode()[0]
print(f"Ev fiyatlarında en sık tekrar eden fiyat (mod): {mod_value}")

# Çarpıklık değeri
skew_value = df['Price'].skew()
print(f"Ev fiyatlarının çarpıklık değeri: {skew_value}")

# Ortalama ve standart sapma
mean_price = df['Price'].mean()
std_price = df['Price'].std()

# Aykırı değer sınırları (3 standart sapma kuralı)
lower_bound = mean_price - (3 * std_price)
upper_bound = mean_price + (3 * std_price)

# Aykırı değerler
outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
print(f"Aykırı değerlerin sayısı: {outliers.shape[0]}")
print(f"Aykırı değerlerin alt sınırı: {lower_bound}")
print(f"Aykırı değerlerin üst sınırı: {upper_bound}")

# Sonsuz değerleri NaN ile değiştir
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Kategorik değişkenleri one-hot encoding ile dönüştürelim
df_encoded = pd.get_dummies(df, drop_first=True)

# Korelasyon matrisini hesaplayalım
correlation_matrix = df_encoded.corr()

# Korelasyon matrisini ısı haritası ile görselleştirelim
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Korelasyon Matrisi Isı Haritası')
plt.show()

# Özellikler (X) ve hedef değişken (y)
X = df_encoded[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']]
y = df_encoded['Price']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli tanımla ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Test setinde tahminler yap
y_pred = model.predict(X_test)

# MSE ve R² değerlerini hesapla
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Ortalama Kare Hatası (MSE):", mse)
print("R Kare Değeri (R²):", r2)

# Alan ile Fiyat İlişkisi
plt.figure(figsize=(12, 6))
sns.barplot(x='Area', y='Price', data=df_encoded)
plt.title('Alan ile Fiyat Arasındaki İlişki')
plt.xlabel('Alan (metrekare)')
plt.ylabel('Fiyat')
plt.grid()
plt.show()

# Yatak Sayısı ile Fiyat İlişkisi
plt.figure(figsize=(12, 6))
sns.barplot(x='Bedrooms', y='Price', data=df_encoded)
plt.title('Yatak Sayısı ile Fiyat Arasındaki İlişki')
plt.xlabel('Yatak Sayısı')
plt.ylabel('Fiyat')
plt.grid()
plt.show()

# Banyo Sayısı ile Fiyat İlişkisi
plt.figure(figsize=(12, 6))
sns.barplot(x='Bathrooms', y='Price', data=df_encoded)
plt.title('Banyo Sayısı ile Fiyat Arasındaki İlişki')
plt.xlabel('Banyo Sayısı')
plt.ylabel('Fiyat')
plt.grid()
plt.show()

# Kat Sayısı ile Fiyat İlişkisi
plt.figure(figsize=(12, 6))
sns.barplot(x='Floors', y='Price', data=df_encoded)
plt.title('Kat Sayısı ile Fiyat Arasındaki İlişki')
plt.xlabel('Kat Sayısı')
plt.ylabel('Fiyat')
plt.grid()
plt.show()

# İnşa Yılı ile Fiyat İlişkisi
plt.figure(figsize=(12, 6))
sns.barplot(x='YearBuilt', y='Price', data=df_encoded)
plt.title('İnşa Yılı ile Fiyat Arasındaki İlişki')
plt.xlabel('İnşa Yılı')
plt.ylabel('Fiyat')
plt.grid()
plt.show()

# 2. Polinom Regresyon (Polynomial Regression)

# Alan ve fiyat verilerini seç
X = df[['Area']].values
y = df['Price'].values

# Veriyi eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polinom özellikleri oluştur
degree = 2  # Polinom derecesi (örneğin 2. derece)
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X_train)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_poly, y_train)

# Test verileri için polinom özellikleri oluştur
X_test_poly = poly_features.transform(X_test)
y_pred = model.predict(X_test_poly)

# MSE ve R² değerlerini hesapla
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ortalama Kare Hatası (MSE): {mse}")
print(f"R Kare Değeri (R²): {r2}")

# Polinom regresyonu grafiği
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Gerçek Veriler')
plt.scatter(X_test, y_pred, color='red', label='Tahmin Edilen Veriler')
plt.title('Polinom Regresyonu')
plt.xlabel('Alan (metrekare)')
plt.ylabel('Fiyat')
plt.legend()
plt.show()
