import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Data
sales_data = pd.read_csv("satis_verisi_5000.csv")
customer_data = pd.read_csv("musteri_verisi_5000_utf8.csv")

# Missing Value Check
print("Sales Missing Value:", sales_data.isnull().sum())
print("Customer Missing Value", customer_data.isnull().sum())

# Outlier Analysis - Fiyat
Q1_fiyat = sales_data['fiyat'].quantile(0.25)
Q3_fiyat = sales_data['fiyat'].quantile(0.75)
IQR_fiyat = Q3_fiyat - Q1_fiyat
lower_bound_fiyat = Q1_fiyat - 1.5 * IQR_fiyat
upper_bound_fiyat = Q3_fiyat + 1.5 * IQR_fiyat
sales_data['fiyat'] = sales_data['fiyat'].clip(lower=lower_bound_fiyat, upper=upper_bound_fiyat)

# Outlier Analysis - Toplam Satış
Q1_toplam_satis = sales_data['toplam_satis'].quantile(0.25)
Q3_toplam_satis = sales_data['toplam_satis'].quantile(0.75)
IQR_toplam_satis = Q3_toplam_satis - Q1_toplam_satis
lower_bound_toplam_satis = Q1_toplam_satis - 1.5 * IQR_toplam_satis
upper_bound_toplam_satis = Q3_toplam_satis + 1.5 * IQR_toplam_satis
sales_data['toplam_satis'] = sales_data['toplam_satis'].clip(lower=lower_bound_toplam_satis, upper=upper_bound_toplam_satis)

# Merge Datasets
merged_data = pd.merge(sales_data, customer_data, on="musteri_id", how="inner")

# Convert Date
merged_data['tarih'] = pd.to_datetime(merged_data['tarih'], errors='coerce')
print("Hatalı Tarihler:", merged_data[merged_data['tarih'].isna()])

# Weekly and Montly Sales Analysis
merged_data['hafta'] = merged_data['tarih'].dt.isocalendar().week
weekly_sales = merged_data.groupby('hafta')['toplam_satis'].sum()

merged_data['ay'] = merged_data['tarih'].dt.month
monthly_sales = merged_data.groupby('ay')['toplam_satis'].sum()

# Plotting

# Aylık Toplam Satış Trendleri
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', title="Aylık Toplam Satış Trendleri", marker='o')
plt.xlabel("Ay")
plt.ylabel("Toplam Satış")
plt.grid()
plt.show()

# Haftalık Toplam Satış Trendleri
plt.figure(figsize=(10, 6))
weekly_sales.plot(kind='line', title="Haftalık Toplam Satış Trendleri", color='orange', marker='o')
plt.xlabel("Hafta")
plt.ylabel("Toplam Satış")
plt.grid()
plt.show()

# Kategori Bazlı Satış Analizi
category_sales = sales_data.groupby('kategori')['toplam_satis'].sum()
print("Kategori Bazlı Toplam Satış:\n", category_sales)

# Yaş Gruplarına Göre Satış Eğilimi
bins = [0, 25, 35, 50, np.inf]
labels = ["18-25", "26-35", "36-50", "50+"]
merged_data['yas_grubu'] = pd.cut(merged_data['yas'], bins=bins, labels=labels, right=False)
sales_by_age = merged_data.groupby('yas_grubu')['toplam_satis'].sum()
print("Yaş Gruplarına Göre Toplam Satış:\n", sales_by_age)

# Şehir Bazlı Harcama Analizi
city_spending = merged_data.groupby('sehir')['harcama_miktari'].sum().sort_values(ascending=False)
print("Şehir Bazlı Harcama:\n", city_spending)

# Aylık Kategorilere Göre Satış Analizi
monthly_category_sales = merged_data.groupby(['kategori', 'ay'])['toplam_satis'].sum().unstack()
monthly_category_sales.plot(kind='line', title="Aylık Kategorilere Göre Satış Trendleri", figsize=(10, 6))
plt.ylabel("Toplam Satış")
plt.show()

# Pareto Analizi
pareto_data = sales_data.groupby('ürün_adi')['toplam_satis'].sum().sort_values(ascending=False)
pareto_data_cumsum = pareto_data.cumsum() / pareto_data.sum()
pareto_cutoff = pareto_data_cumsum[pareto_data_cumsum <= 0.8]
print("Pareto (%80 Satış):\n", pareto_cutoff)


