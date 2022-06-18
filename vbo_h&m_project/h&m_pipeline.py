import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.options.display.max_rows = 999


def prepare_data():
    # Read Data
    articles_df = pd.read_csv("data/articles.csv")
    customer_df = pd.read_csv("data/customers.csv")
    transactions_df = pd.read_csv("data/transactions_train.csv")

    # Articles dataframe hazırla.

    # Eksik ürün açıklamalarını düşür. (416)
    articles_df.dropna(inplace=True)

    # Customer dataframe hazırla.

    # NaN değerleri 0 ile doldur FN (895050) & Active (907576) özellikleri için.
    customer_df["FN"].fillna(0, inplace=True)
    customer_df["Active"].fillna(0, inplace=True)

    # fashion_news_frequency None değerlerini NONE yap (2).
    customer_df.loc[customer_df["fashion_news_frequency"] == "None", ["fashion_news_frequency"]] = "NONE"

    # fashion_news_frequency de NaN değerlerini NONE ile değiştir (16009)
    customer_df["fashion_news_frequency"].fillna("NONE", inplace=True)

    # Age değişkeninde NaN değerlerini median ile doldur (15861).
    customer_df["age"].fillna(customer_df["age"].median(), inplace=True)

    # club_member_status de NaN değerlerini NONE ile doldur (6062).
    customer_df["club_member_status"].fillna("NONE", inplace=True)

    # Transaction dataframe hazırla.

    # Skewed price değişkenine log transformation uygula (3.110 to -0.399).
    transactions_df["price"] = np.log(transactions_df["price"])

    # transactions verisetinde bulunan cutomer_id değerlerini hexadecimal değerlerden integer değerlere çevir. Böylece
    # işlemler daha hızlı olur.
    transactions_df['customer_id'] = transactions_df['customer_id'].str[-16:].apply(int, base=16)

    # Aynı şekilde article_id değerlerinide int32 ye çevirebiliriz. Int32 var olan değer aralığını karşılayabilir.
    transactions_df['article_id'] = transactions_df.article_id.astype('int32')

    # Tarihlerin cinsini datetime cinsine dönüştür.
    transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)

    print("Finished Data Preparation")
    return transactions_df, customer_df, articles_df


# Find Article Pairs
def purchased_together():
    import cudf, gc
    import cv2, matplotlib.pyplot as plt
    from os.path import exists

    df = cudf.read_csv('data/transactions_train.csv')

    # Dataframe bellkete kapladığı alanı düşürelim.
    df = df[['customer_id', 'article_id']]
    df.customer_id = df.customer_id.str[-16:].str.hex_to_int().astype('int64')
    df.article_id = df.article_id.astype('int32')
    _ = gc.collect()

    vc = df.article_id.value_counts()
    pairs = {}
    for j, i in enumerate(vc.index.values):
        if j % 10 == 0: print(j, ', ', end='')
        USERS = df.loc[df.article_id == i.item(), 'customer_id'].unique()
        vc2 = df.loc[(df.customer_id.isin(USERS)) & \
                     (df.article_id != i.item()), 'article_id']
        if len(vc2) == 0:
            continue
        vc2 = vc2.value_counts()
        pairs[i.item()] = vc2.index[0]

        # dosya olarak kaydedelim.
        np.save('pairs_cudf.npy', pairs)

        return pairs


# Prepare Submission File
def prep_submission():
    # Hazırlanmış verisetlerini al
    transactions_df, customer_df, articles_df = prepare_data()

    # Bu modelimiz için ihtiyacımız olan özellikleri transaction veri setinden seçelim.
    transactions_df = transactions_df[['t_dat', 'customer_id', 'article_id']]

    # Müşterilerin geçen hafta yaptığı işlemleri alalım.
    tmp = transactions_df.groupby('customer_id').t_dat.max().reset_index()
    # Her bir müşteri için son işlem tarihi
    tmp.columns = ['customer_id', 'max_dat']

    # Müşterilerin son işlem tarihlerini transaction_df ekleyelim.
    transactions_df = transactions_df.merge(tmp, on=['customer_id'], how='left')
    # Son işlem tarihi ile o satırın işlem tarihlerinin farklarını alıp güne çevirelim.
    transactions_df['diff_dat'] = (transactions_df.max_dat - transactions_df.t_dat).dt.days
    # Müşterilerin son hafta yaptığı işlemleri seçelim.
    transactions_df = transactions_df.loc[transactions_df['diff_dat'] <= 6]

    # Müşterilerin son hafta en çok satın aldıkları ürünleri belirleyelim.

    # Her bir müşteri için son hafta en çok alınan ürünü belirleyelim.
    tmp = transactions_df.groupby(['customer_id', 'article_id']).agg({'t_dat': 'count'}).reset_index()
    tmp.columns = ['customer_id', 'article_id', 'product_count']

    # Son hafta en sık alınan ürünü belirle.
    transactions_df = transactions_df.merge(tmp, on=['customer_id', 'article_id'], how='left')
    transactions_df = transactions_df.sort_values(['product_count', 't_dat'], ascending=False)

    # Bir üründen 100 defa alınmış ise o satırdan 100 adet bulunuyor onları kaldıralım.
    transactions_df = transactions_df.drop_duplicates(['customer_id', 'article_id'])

    # Sıralı bir şekilde transactions_df alalım.
    transactions_df = transactions_df.sort_values(['product_count', 't_dat'], ascending=False)

    # Beraber satın alınan ürünleri belirle.

    # purchased_together() methodunu kullanarak pair çiftini bulabiliriz. Bu method gpu kullanılarak çalıştırılıp
    # sonuçları pairs_cudf.npy olarak kaydedildi.
    # pairs = purchased_together()

    # pair larımızı dosyadan çekiyoruz.
    pairs = np.load('data/pairs_cudf.npy', allow_pickle=True).item()

    # pair ları transaction_df kaydedelim.
    transactions_df['article_id2'] = transactions_df.article_id.map(pairs)

    # customer_id ler ile article_id2 leri ayrı alalım. Bunu ileride birleştireceğiz.
    train2 = transactions_df[['customer_id', 'article_id2']].copy()
    train2 = train2.loc[train2.article_id2.notnull()]
    train2 = train2.drop_duplicates(['customer_id', 'article_id2'])
    train2 = train2.rename({'article_id2': 'article_id'}, axis=1)

    # article_id2 ile oluşturduğumuz train2 setini transaction_df dataframe ine ekleyelim.
    transactions_df = transactions_df[['customer_id', 'article_id']]
    transactions_df = pd.concat([transactions_df, train2], axis=0, ignore_index=True)
    transactions_df.article_id = transactions_df.article_id.astype('int32')
    transactions_df = transactions_df.drop_duplicates(['customer_id', 'article_id'])

    # toplamda 12 ürün önerisi yapacağız ve bunları bir string olarak belirteceğiz. Bunun için elimizde olan article_id
    # değerlerini birleştirelim.
    transactions_df.article_id = ' 0' + transactions_df.article_id.astype('str')
    preds = pd.DataFrame(transactions_df.groupby('customer_id').article_id.sum().reset_index())
    preds.columns = ['customer_id', 'prediction']

    # Geçen haftanın en popüler 12 ürünü getir.
    train = pd.read_csv('data/transactions_train.csv')
    train['customer_id'] = train['customer_id'].str[-16:].str[-16:].apply(int, base=16)
    train['article_id'] = train.article_id.astype('int32')
    train.t_dat = pd.to_datetime(train.t_dat)
    train = train[['t_dat', 'customer_id', 'article_id']]
    train.t_dat = pd.to_datetime(train.t_dat)
    train = train.loc[train.t_dat >= pd.to_datetime('2020-09-16')]
    top12 = ' 0' + ' 0'.join(train.article_id.value_counts().index.astype('str')[:12])

    print("Finish Prep Submission ...")
    return top12, preds


def create_submission_csv():
    top12, preds = prep_submission()
    sub = pd.read_csv('data/sample_submission.csv')
    sub = sub[['customer_id']]
    sub['customer_id_2'] = sub['customer_id'].str[-16:].str[-16:].apply(int, base=16)
    sub = sub.merge(preds.rename({'customer_id': 'customer_id_2'}, axis=1), \
                    on='customer_id_2', how='left').fillna('')
    del sub['customer_id_2']
    sub.prediction = sub.prediction + top12
    sub.prediction = sub.prediction.str.strip()
    sub.prediction = sub.prediction.str[:131]
    sub.to_csv(f'submission.csv', index=False)
    sub.head()


if __name__ == '__main__':
    create_submission_csv()
