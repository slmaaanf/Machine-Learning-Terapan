# Laporan Proyek Machine Learning - Salma Nurfauziah

## Project Overview

![Books](https://github.com/user-attachments/assets/44b105d4-843b-48b0-932e-8812eb447fc8)

Beragamnya jenis buku dan banyaknya jumlah buku merupakan permasalahan tersendiri bagi para pembaca buku. Salah satu permasalahan yang muncul adalah saat pembaca kesulitan dalam menentukan buku yang akan dibaca selanjutnya. Sistem rekomendasi buku sangat penting dalam era digital karena membantu pengguna menemukan buku yang sesuai dengan preferensi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis Content-Based Filtering dan Collaborative Filtering dengan dataset dari Kaggle.

💡 Manfaat Proyek:

✔ Membantu pengguna menemukan buku yang sesuai dengan minatnya.

✔ Meningkatkan pengalaman pengguna dalam memilih buku.

✔ Menggunakan metode berbasis konten dan kolaboratif untuk rekomendasi yang lebih akurat.

  Format Referensi: [SISTEM REKOMENDASI BUKU](http://eprints.undip.ac.id/65823/) 

## Business Understanding

📝 Problem Statements
Bagaimana sistem rekomendasi dapat membantu pengguna menemukan buku yang sesuai dengan preferensinya?
Seberapa efektif metode Content-Based Filtering dibandingkan dengan Collaborative Filtering dalam merekomendasikan buku?

🎯 Goals
Mengembangkan sistem rekomendasi yang dapat menyarankan buku berdasarkan preferensi pengguna.
Membandingkan performa Content-Based Filtering dan Collaborative Filtering untuk memahami pendekatan terbaik.

🛠 Solution Approach

✔ Content-Based Filtering: Menggunakan informasi deskriptif dari buku untuk memberikan rekomendasi.

✔ Collaborative Filtering: Menggunakan interaksi pengguna dan rating untuk merekomendasikan buku.

## Data Understanding
Dataset yang digunakan diperoleh dari Kaggle dan terdiri dari tiga tabel utama [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?).

🔍 Eksplorasi Data

✔ Memeriksa jumlah total pengguna, buku, dan rating.

✔ Menganalisis distribusi rating untuk melihat pola evaluasi buku oleh pengguna.

✔ Visualisasi hubungan antara pengguna dan buku berdasarkan rating.

```
books = pd.read_csv(dataset_path + "/Books.csv")
users = pd.read_csv(dataset_path + "/Users.csv")
ratings = pd.read_csv(dataset_path + "/Ratings.csv")
```

📂 **Dataset Components:**
| Dataset  | Jumlah Data  | Fitur  |
|----------|-------------|-----------------|
| Users    | 278.858     | User-ID, Location, Age |
| Ratings  | 340.556     | User-ID, ISBN, Book-Rating |
| Books    | 271.360     | ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher |

📌 **Uraian Fitur:**
- **ISBN**: Kode unik untuk setiap buku.
- **Book-Title**: Judul buku.
- **Book-Author**: Nama penulis buku.
- **Year-Of-Publication**: Tahun terbit buku.
- **Publisher**: Nama penerbit buku.
- **User-ID**: ID unik pengguna.
- **Book-Rating**: Rating buku dari pengguna (rentang 1-10).

Dari hasil jumlah data yang didapat tidak terdapat duplikat tetapi adanya missing value pada data. 

:books: Books 

![image](https://github.com/user-attachments/assets/a83cfdc0-29bc-4026-84cd-b4608b19622f)

:chart_with_downwards_trend: Ratings

![image](https://github.com/user-attachments/assets/a42bdce6-2f31-4718-bb5a-3b29cb9bd893)

:bust_in_silhouette: Users

![image](https://github.com/user-attachments/assets/6879cf64-39bf-41af-a1ee-5363e10238c6)

### Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami pola distribusi data, jumlah pengguna, buku, serta distribusi rating. 
- 10 penulis dengan jumlah buku terbanyak dianalisis, dengan Agatha Christie sebagai penulis teratas.

![image](https://github.com/user-attachments/assets/a4250ae2-516d-4d95-97a1-2cdf9fc2c21b)

- Distribusi rating menunjukkan mayoritas nilai berada dalam rentang tertentu.

:pushpin: Rating Buku

  ![image](https://github.com/user-attachments/assets/dd76d610-f95b-489b-99df-b5758115b5aa)

  :pushpin: Rating per Pengguna
  
  ![image](https://github.com/user-attachments/assets/ce8ca0ff-1bca-4d2f-aaf6-d44368b18ac3)

  Hasil EDA ini membantu dalam mempersiapkan data sebelum pemodelan. 🚀

## Data Preparation
✔ **Cleaning Data:** Menghapus nilai yang hilang dan duplikasi.
✔ **Menggabungkan Data:** Menggabungkan data rating dengan book berdasarkan ISBN.
✔ **Menangani Missing Values:** Mengecek dan menghapus nilai yang hilang setelah penggabungan.
✔ **Memfilter Data:** Memilih hanya buku dengan jumlah rating cukup (minimal 5 review).
✔ **Hasil Akhir Data:** Menampilkan dataset yang sudah dibersihkan.

1️⃣ Cleaning Data (Menghapus duplikasi & nilai yang hilang)

```
# Menghapus duplikasi
books.drop_duplicates(inplace=True)

# Mengecek jumlah nilai yang hilang
print("Missing Values sebelum dibersihkan:\n", books.isnull().sum())

# Menghapus baris dengan missing values jika perlu
books.dropna(inplace=True)

print("Missing Values setelah dibersihkan:\n", books.isnull().sum())
print("Data setelah menghapus missing values:", books.shape)

```
:pushpin: Terdapat missing value di bagian Book-Author dan Publisher.

![image](https://github.com/user-attachments/assets/39a96833-92ae-4b13-a27d-ea3cc687914c)

2️⃣ Menggabungkan Data (Rating + Books)

```
# Menggabungkan dataframe ratings dengan books berdasarkan nilai ISBN
books = pd.merge(ratings, books, on='ISBN', how='left')
books
```

![image](https://github.com/user-attachments/assets/172dcad9-8b34-4e73-8789-7f8dbbe98cb2)

3️⃣ Menangani Missing Values Setelah Penggabungan

```
books.isnull().sum()  # Mengecek jumlah missing values
books.dropna(inplace=True)  # Menghapus baris dengan missing values jika perlu
# Menampilkan jumlah missing values di setiap kolom
print("Missing Values:\n", books.isnull().sum())
print("Data setelah menghapus missing values:", books.shape)
```

![image](https://github.com/user-attachments/assets/066054ec-9cbb-4df1-878b-45a11881ea25)

4️⃣ Memfilter Data dengan Rating Cukup

```
book_counts = books.groupby('ISBN')['Book-Rating'].count()
books = books[books['ISBN'].isin(book_counts[book_counts >= 5].index)]
# Menghitung jumlah rating per ISBN

print("Jumlah data setelah memfilter buku dengan minimal 5 rating:", books.shape)

```
``
Jumlah data setelah memfilter buku dengan minimal 5 rating: (670480, 7)
``

5️⃣ Hasil Data Setelah Dibersihkan

![image](https://github.com/user-attachments/assets/a96c2380-cccb-4491-929a-c58277fc861f)

6️⃣ Jumlah data setelah di bersihkan

```
## Data Preparation

# Menghapus rating 0 karena tidak merepresentasikan penilaian
ratings = ratings[ratings['Book-Rating'] > 0]

print("\nJumlah data setelah pembersihan:")
print(f"Books: {books.shape[0]} baris")
print(f"Users: {users.shape[0]} baris")
print(f"Ratings: {ratings.shape[0]} baris")
```
``
Jumlah data setelah pembersihan:
Books: 670480 baris
Users: 278858 baris
Ratings: 433671 baris
``

:pushpin: Kesimpulan

- Tahapan data preparation berhasil membersihkan dataset dengan menghapus duplikasi, menangani nilai yang hilang, dan memastikan hanya buku dengan jumlah rating yang memadai digunakan dalam model. Hal ini memastikan bahwa model rekomendasi akan bekerja dengan data yang lebih akurat dan relevan.
  
## Modeling
Pada bagian ini dibagi menjadi 2 tahap model yaitu :

:sparkles: content based filtering

:sparkles: collaborative filtering

📖 Content-Based Filtering
- Menggunakan Nearest Neighbors untuk mencari buku serupa berdasarkan fitur judul buku.

```
## Content-Based Filtering

tfidf = TfidfVectorizer(stop_words='english') # Initialize TfidfVectorizer
tfidf_matrix = tfidf.fit_transform(books['Book-Title']) # Fit and transform book titles

nn = NearestNeighbors(metric='cosine', algorithm='brute') # Initialize NearestNeighbors
nn.fit(tfidf_matrix) # Fit the TF-IDF matrix

```
![image](https://github.com/user-attachments/assets/6bf7da17-8858-4861-ab50-518311c88ae9)


🔎 Collaborative Filtering

Pendekatan ini menggunakan metode Singular Value Decomposition (SVD) untuk mendekomposisi matriks pengguna-buku dan memprediksi rating.

```
## Collaborative Filtering (Matrix Factorization)
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
model = SVD()
cross_validate(model, data, cv=5, verbose=True)
```

![image](https://github.com/user-attachments/assets/3506a2a8-4711-4237-87e9-b0fc53756bf6)


### **Kelebihan & Kekurangan**
| Pendekatan                | Kelebihan                                    | Kekurangan                                    |
|---------------------------|----------------------------------------------|----------------------------------------------|
| **Content-Based Filtering** | Tidak memerlukan data pengguna lain | Terbatas pada metadata yang tersedia |
| **Collaborative Filtering** | Mampu memberikan rekomendasi personal | Membutuhkan data interaksi yang cukup |

## Evaluation
Untuk mengevaluasi model, digunakan metrik RMSE serta Precision@K dan Recall@K.
- RMSE mengukur seberapa dekat prediksi dengan rating asli.
- Precision@K dan Recall@K mengevaluasi relevansi rekomendasi yang diberikan.

:boom: Menggunakan Precision@K dan Recall@K untuk evaluasi Content-Based Filtering

```
## Collaborative Filtering
from collections import defaultdict
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
model = SVD()
cross_validate(model, data, cv=5, verbose=True)

# Fit model and generate predictions
trainset = data.build_full_trainset()
model.fit(trainset)
predictions = model.test(trainset.build_testset()) # Generate predictions for evaluation


#Evaluasi Content-Based Filtering
def precision_recall_at_k(predictions, k=5, threshold=7):
    """Menghitung Precision@K dan Recall@K"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)
        precision = n_rec_k / k if k else 0
        recall = n_rec_k / n_rel if n_rel else 0
        precisions.append(precision)
        recalls.append(recall)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)
precision, recall = precision_recall_at_k(predictions, k=5, threshold=7)
print(f'Precision@5: {precision:.4f}')
print(f'Recall@5: {recall:.4f}')
```
![image](https://github.com/user-attachments/assets/d8cd15b7-a62a-4e29-bdcb-ad240c4e9f75)

:boom: RMSE untuk Collaborative Filtering 

💡 Formula RMSE:

![image](https://github.com/user-attachments/assets/a05c80de-f628-420f-9d0b-f699982a97ae)

```
#Evaluasi RMSE untuk Collaborative Filtering dan Neural Network

from sklearn.metrics import mean_squared_error
trainset = data.build_full_trainset()
model.fit(trainset)
predictions = model.test(trainset.build_testset())
rmse = np.sqrt(mean_squared_error(ratings['Book-Rating'], [pred.est for pred in predictions]))
print("RMSE Score:", rmse)
```
![image](https://github.com/user-attachments/assets/490a821f-59af-4b4d-8a27-4ea34d8ee9de)

Dengan evaluasi ini, kita dapat menilai apakah model sudah memberikan rekomendasi yang baik dan menentukan pendekatan mana yang lebih optimal.

:boom: Neural Network-Based Recommender System

Model ini menggunakan pendekatan berbasis Neural Network untuk memberikan rekomendasi buku berdasarkan interaksi pengguna.

Langkah-langkah yang dilakukan:

1️⃣ Mapping Data ke Indeks Numerik

User-ID dan ISBN dikonversi menjadi indeks integer agar bisa digunakan dalam embedding.

2️⃣ Split Data untuk Training dan Validation

Dataset dibagi menjadi 80% training dan 20% validation menggunakan train_test_split().

3️⃣ Membangun Arsitektur Neural Network

Input layer untuk user dan buku.
Embedding layer untuk membuat representasi vektor masing-masing.
Dense layers dengan ReLU activation untuk menangkap hubungan kompleks.
Output layer untuk memprediksi rating buku.

4️⃣ Melatih Model

Model dikompilasi dengan Adam optimizer dan Mean Squared Error (MSE) sebagai loss function.
Dilatih selama 10 epoch dengan batch size 32.

5️⃣ Evaluasi Model

RMSE digunakan untuk mengukur error antara prediksi dan rating asli.
Jika train RMSE terus menurun tetapi validation RMSE tetap tinggi, berarti model mengalami overfitting.

```
# Prepare data for the neural network
user_ids = ratings['User-ID'].unique()
book_isbns = ratings['ISBN'].unique()

user_mapping = {user_id: index for index, user_id in enumerate(user_ids)}
book_mapping = {isbn: index for index, isbn in enumerate(book_isbns)}

ratings['User-ID'] = ratings['User-ID'].map(user_mapping)
ratings['ISBN'] = ratings['ISBN'].map(book_mapping)

num_users = len(user_ids)
num_books = len(book_isbns)

X = ratings[['User-ID', 'ISBN']].values
y = ratings['Book-Rating'].values
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model using the functional API
user_input = keras.Input(shape=(1,), name='user_id')
book_input = keras.Input(shape=(1,), name='isbn')

user_embedding = layers.Embedding(num_users, 50, input_length=1)(user_input)
book_embedding = layers.Embedding(num_books, 50, input_length=1)(book_input)

merged = layers.concatenate([user_embedding, book_embedding], axis=1)
flattened = layers.Flatten()(merged)
dense1 = layers.Dense(128, activation='relu')(flattened)
dense2 = layers.Dense(64, activation='relu')(dense1)
output = layers.Dense(1)(dense2)

model = keras.Model(inputs=[user_input, book_input], outputs=output)


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])


history = model.fit(
    x=[x_train[:, 0], x_train[:, 1]], y=y_train,
    epochs=10,
    batch_size=32,
    validation_data=([x_val[:, 0], x_val[:, 1]], y_val)
)
```
![image](https://github.com/user-attachments/assets/a2f62b7b-5c65-4092-93b3-8149838b5016)

💡 Visualisasi Evaluasi

Berdasarkan grafik RMSE:
✔ **Train RMSE menurun secara signifikan**, menunjukkan model belajar dengan baik pada data pelatihan.

✔ **Validation RMSE tetap tinggi dan cenderung stabil**, menunjukkan kemungkinan **overfitting**.

✔ Perlu dilakukan regularisasi tambahan atau tuning hyperparameter agar model tidak terlalu menyesuaikan data pelatihan.

```
#Visualisasi Evaluasi
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
plt.plot(history.history['root_mean_squared_error'], marker="o", linestyle="-", color="b", label="Train RMSE")
plt.plot(history.history['val_root_mean_squared_error'], marker="s", linestyle="--", color="r", label="Validation RMSE")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title("Evaluasi RMSE Model")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/38801c17-0b59-402b-a0f2-3c5c336c5a80)

# Kesimpulan

Dari proyek ini, dapat disimpulkan bahwa:

:star2: Content-Based Filtering cocok untuk rekomendasi berbasis metadata buku.

:star2: Collaborative Filtering lebih efektif dalam memberikan rekomendasi yang personal.

:star2: Neural Network-Based Recommender System menghasilkan hasil terbaik dalam memahami hubungan kompleks antara pengguna dan buku.

Proyek ini memberikan wawasan mengenai perbandingan dua metode rekomendasi dan dapat dikembangkan lebih lanjut dengan menggabungkan metode untuk meningkatkan akurasi.

