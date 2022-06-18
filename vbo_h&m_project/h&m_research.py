# H&M Personalized Fashion Recommendations

# Dosyalar:
# articles.csv : Ürünler ile ilgili metadata içerir.
# customer.csv : Müşteriler ile ilgili metadata içerir.
# sample_submission.csv : Gönderim formatı. Tahmin edilecek müşteriler.
# transaction_train.csv : Müşterinin yaptığı ödemeleri içerir. Duplicate satırlar müşterinin
# aynı ürün için birden fazla kez ödeme yaptığını gösterir.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.options.display.max_rows = 999


def check_df(df, percentiles=None):
    print("----------------------------- Head ------------------------------\n")
    display(df.head())
    print("----------------------------- Tail ------------------------------\n")
    display(df.tail())
    print("\n----------------------------- Shape -----------------------------\n")
    print(df.shape)
    print("\n----------------------------- Types -----------------------------\n")
    print(df.dtypes)
    print("\n----------------------------- NaN -------------------------------\n")
    print(df.isnull().sum())
    print("\n----------------------------- Quantiles -------------------------")
    display(df.describe(percentiles=percentiles).T)


def category_summary(df, col_names, plot=False, horizontal=False, title=None):
    for col_name in col_names:
        new_df = pd.DataFrame({col_name: df[col_name].value_counts(),
                               "Ratio": 100 * df[col_name].value_counts() / len(df)})
        display(new_df)
        if plot:
            if horizontal:
                sns.countplot(y=col_name, data=df, order=df[col_name].value_counts().index)
                plt.title(title)
                plt.show()
            else:
                sns.countplot(x=col_name, data=df, order=df[col_name].value_counts().index)
                plt.title(title)
                plt.show()


def category_sub_grup(df, col_name, sub_group, title=None):
    f, ax = plt.subplots(figsize=(15, 7))
    ax = sns.histplot(data=df, y=col_name, color='orange', hue=sub_group, multiple="stack")
    ax.set_xlabel('count')
    ax.set_ylabel(col_name)
    plt.title(title)
    plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False, q1=0.25, q3=0.75):
    low, up = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0])
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def write_custom_box_plot(df, col_name, q1=25, q3=75, title=None):
    import matplotlib.cbook as cbook

    stats = {}
    stats[col_name] = cbook.boxplot_stats(df[col_name])[0]
    stats[col_name]['q1'], stats[col_name]['q3'] = np.percentile(df[col_name], [q1, q3])

    fig, ax = plt.subplots(1, 1)
    plt.title(title)
    ax.bxp([stats[col_name]], positions=range(1))


# 1. Articles EDA

# article_id : Ürünlerin benzersiz ayırt edici özelliğidir.
# product_code, prod_name : Ürünlerin isimleri ve kodları.
# product_type, product_type_name : Ürün cinslerinin ismleri ve kodları
# graphical_appearance_no, graphical_appearance_name : Ürünlerin görünüş şekillerinin ismileri ve kodları.
# colour_group_code, colour_group_name : Ürünlerin renk grupları ve kodları
# perceived_colour_value_id, perceived_colour_value_name: Algılanan ürün renkleri
# perceived_colour_master_id, perceived_colour_master_name : Algılanan ürün renkleri
# department_no, department_name: : Ürünlerin ait olduğu departmanların no ları ve isimleri
# index_code, index_name: : Ürünlerin indeks kodları ve isimleri.
# index_group_no, index_group_name: : İndeks gruplarının no ları ve isimleri
# section_no, section_name: : Bölümlerin no ları ve isimleri
# garment_group_no, garment_group_name: : Giysi gruplarının numaraları ve isimleri
# detail_desc: : Giysilerin açıklamaları


articles_df = pd.read_csv("data/articles.csv")

# tüm dataframe incele
check_df(articles_df)

# product_group_no feature incele
category_summary(articles_df, ["product_group_name"], plot=True, horizontal=True, title="H&M Ürün Grubları")

# garment_group_no, garment_group_name
category_summary(articles_df, ["garment_group_name"], plot=True, horizontal=True, title="H&M Giysi Grubları")

# index_group_no, index_group_name
category_summary(articles_df, ["index_group_name"], plot=True, horizontal=True, title="H&M Index Grubları")

# H&M Ürünlerin İndekslere Göre Dağılımları
category_sub_grup(articles_df, "product_group_name", "index_group_name", "H&M Ürünlerin İndekslere Göre Dağılımları")
articles_df.groupby(["product_group_name", "index_group_name"]).agg({"index_group_name": ["count"]})

# H&M Giysilerin İndekslere Göre Dağılımları
category_sub_grup(articles_df, "garment_group_name", "index_group_name", "H&M Giysilerin İndekslere Göre Dağılımları")
articles_df.groupby(["garment_group_name", "index_group_name"]).agg({"index_group_name": ["count"]})

# İndeks Kodlarının, İndeks Grupları İçerisindeki Dağılımları
category_sub_grup(articles_df, "index_group_name", "index_name", "H&M İndeks Kodlarının, İndeks Grupları İçerisindeki "
                                                                 "Dağılımları")
articles_df.groupby(["index_group_name", "index_name"]).agg({"index_name": ["count"]})

# product_type_no & product_type_name
category_summary(articles_df, ["product_type_name"], plot=True)

# En Çok Bulunan 20 Ürün Cinsi
top_20_product_types = articles_df.groupby(["product_type_name"]).count().article_id.sort_values(ascending=False)[:20]
top_20_product_types = pd.DataFrame(top_20_product_types)
top_20_product_types["ratio"] = (top_20_product_types["article_id"] / len(articles_df)) * 100
display(top_20_product_types)

# Transactions EDA

# t_dat: Her bir ürün için yapılan işelmin tarihidir. Dikkat: örneğin şoktan alışveriş yaptın ve sana bir fatura verildi
# ya elimizdeki veri setinde o faturada bulunan her bir ürün için bir  row bulunur. Buna ek olarak eğer aynı ürün
# birden fazla kez satın alınmış ise bunlarda row olarak belirtilir. Bunlar bizim için duplicate row olur.
# customer_id: Müşteri id'leri
# article_id: Ürün id'leri
# price: İşlemde ödenen tutar
# sales_channel_id: satış kanalı


transactions_df = pd.read_csv("data/transactions_train.csv")

check_df(transactions_df)

# Müşteri başına kaç işlem var onaq bakalım
transactions_df.groupby('customer_id').count()

# Minimum ve maksimum işlem tarihlerine bakalım
print(transactions_df["t_dat"].min())
print(transactions_df["t_dat"].max())

# Kaç tane farklı gün var
print(transactions_df["t_dat"].nunique())

# Price değişkeninde outlier incele
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=transactions_df["price"])
plt.show()

# Aykırı değer sayısına bak
grab_outliers(transactions_df, "price", q1=0.25, q3=0.75)

# Aralığı değiştirip aykırı değerleri incele
write_custom_box_plot(df=transactions_df, col_name='price', q1=1, q3=99, title="Price With Custom Q1=0.01, Q3=0.99")
grab_outliers(transactions_df, "price", q1=0.01, q3=0.99)

# Price skewness incelemesi
print(transactions_df["price"].skew())

fig, ax = plt.subplots(figsize=(15, 5))
sns.displot(transactions_df["price"])
ax.set_title('Sale Price Distribution', fontsize=25, loc='center')
ax.set_xlabel('Sale Price', fontsize=23)
plt.tick_params(axis='x', which='major', labelsize=12)
ax.yaxis.tick_left()
plt.show()

# Sales channel id
category_summary(transactions_df, ["sales_channel_id"], plot=True)

# Cutomer EDA

# Notlar

# FN vee active arasında bir korelasyon bulunuyor.
# FN ve Active değerlerinde çok fazla NaN değerleri bulunuyor. İkisinde de 1 ve NaN var sadece.
# Bu NaN değerlerini 1 ve 0 olarak değiştirebiliriz.
# postal kode çok bir anlam ifade etmiyor onu kaldırabiliriz.
# Club_member_status 6062 adet null değeri bulunmaktadır. Club_Member Status nan iken FN ninde nan olduğu 5741 tane
# nan var. fashion news 5740 adet nan ve None değeri var, Active 5744 adet nan. Hepsinin kesişiminde 5734 adet gözlem
# var yani 3 column nan iken club memberlarda 5734 adeti nan dır. Ancak 3 ünün none olduğu değer sayısı çok fazla
# çoğu club member null buraya düşsede diğer değerlerinde çoğu burada olacak. NaN değerleri ile PRE-CREATE arasında
# FN ve Active açısndan mean benzerliği var.
# fashion_news_frequency, None 2 tane silinebilinir. None ile NaN değerlerini eşitlenebilinir.
# Yaşları satın alınan ürünlere göre doldur. Yaş için ciddi bir outlier problemi yok.

customer_df = pd.read_csv("data/customers.csv")

check_df(customer_df)

# FN & Active NaN değerleri incele
customer_df["FN"].value_counts(dropna=False)
customer_df["Active"].value_counts(dropna=False)

# FN ve Active değerlerinin arasındaki ilişkiyi incele.
customer_df[(customer_df["Active"].isnull()) & (customer_df["FN"].isnull())]
customer_df[(customer_df["Active"].isnull()) & (~customer_df["FN"].isnull())]

# club_member_status özelliğini incele
customer_df["club_member_status"].value_counts(dropna=False)

# club_member_status ile FN çaprazla
customer_df[(customer_df["club_member_status"].isnull()) & (customer_df["FN"].isnull())]

# club_member_status ile fashion_news_frequency değerini karşılaştır.
customer_df[(customer_df["club_member_status"].isnull()) & (
        (customer_df["fashion_news_frequency"].isnull()) | (customer_df["fashion_news_frequency"] == 'NONE'))]

# club_member_status ile active karşılaştırılması
customer_df[(customer_df["club_member_status"].isnull()) & ((customer_df["Active"].isnull()))]

# club_member_status değeri için diğer değerlerin ortalmalarına bakalım. Bunun için FN ve Active de null olan değerleri
# 0 yapacağız.
new_customer_df = customer_df
new_customer_df["FN"] = customer_df["FN"].fillna(0)
new_customer_df["Active"] = customer_df["Active"].fillna(0)
new_customer_df.groupby("club_member_status", dropna=False).mean()

# fashion_news_frequency incele
customer_df["fashion_news_frequency"].value_counts(dropna=False)

# age incele
quantiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
display(pd.DataFrame(customer_df["age"].describe(quantiles)).T)

sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10, 5))
ax = sns.histplot(data=customer_df, x='age', bins=50, color='orange')
ax.set_xlabel('Distribution of the customers age')
plt.show()

# age aykırı değerleri incele
grab_outliers(customer_df, "age", q1=0.25, q3=0.75)
ax = sns.boxplot(x=customer_df["age"])
plt.show()
