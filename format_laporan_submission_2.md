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

📂 Dataset Components:

- **Users**: Berisi informasi tentang pengguna, seperti User-ID, Location dan Age.
- **Ratings**: Menyimpan rating yang diberikan oleh pengguna terhadap buku, seperti User-ID, ISBN dan	Book-Rating.
- **Books**: Berisi informasi detail mengenai buku, seperti ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M dan Image-URL-L.

🔍 Eksplorasi Data

✔ Memeriksa jumlah total pengguna, buku, dan rating.

✔ Menganalisis distribusi rating untuk melihat pola evaluasi buku oleh pengguna.

✔ Visualisasi hubungan antara pengguna dan buku berdasarkan rating.

```
books = pd.read_csv(dataset_path + "/Books.csv")
users = pd.read_csv(dataset_path + "/Users.csv")
ratings = pd.read_csv(dataset_path + "/Ratings.csv")
```

:books: Books

![image](https://github.com/user-attachments/assets/a83cfdc0-29bc-4026-84cd-b4608b19622f)

:chart_with_downwards_trend: Ratings

![image](https://github.com/user-attachments/assets/a42bdce6-2f31-4718-bb5a-3b29cb9bd893)

:bust_in_silhouette: Users

![image](https://github.com/user-attachments/assets/6879cf64-39bf-41af-a1ee-5363e10238c6)


### Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami pola distribusi data dalam dataset yang digunakan. Analisis ini melibatkan eksplorasi jumlah total pengguna, buku, serta distribusi rating yang diberikan. Dengan memahami pola ini, kita dapat mengetahui tren umum dalam data, seperti bagaimana pengguna memberikan rating atau seberapa banyak interaksi yang terjadi antara pengguna dan buku.

Hasil dari EDA menunjukkan distribusi rating yang mayoritas berada pada nilai tertentu, serta pola jumlah rating yang diberikan oleh masing-masing pengguna. Informasi ini membantu dalam mempersiapkan data sebelum proses pemodelan.

:pushpin: Rating Buku

  ![image](https://github.com/user-attachments/assets/dd76d610-f95b-489b-99df-b5758115b5aa)

  :pushpin: Rating per Pengguna
  
  ![image](https://github.com/user-attachments/assets/ce8ca0ff-1bca-4d2f-aaf6-d44368b18ac3)

## Data Preparation
Sebelum membangun model, kita harus melakukan pembersihan dan pemrosesan data.

✔ Cleaning Data: Menghapus nilai yang hilang dan duplikasi.

✔ Feature Engineering: Menyiapkan fitur yang relevan untuk model.

✔ Normalization: Melakukan normalisasi data jika diperlukan.

```
## Data Preparation
# Remove duplicates and missing values
books.drop_duplicates(inplace=True)
users.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

# Menghapus rating 0 karena tidak merepresentasikan penilaian
ratings = ratings[ratings['Book-Rating'] > 0]

print("\nJumlah data setelah pembersihan:")
print(f"Books: {books.shape[0]} baris")
print(f"Users: {users.shape[0]} baris")
print(f"Ratings: {ratings.shape[0]} baris")

```

## Modeling
Dua pendekatan utama yang digunakan dalam proyek ini:

📖 Content-Based Filtering

- Menggunakan TF-IDF untuk merepresentasikan judul buku sebagai vektor.
- Menggunakan NearestNeighbors untuk menemukan buku yang mirip berdasarkan judulnya.

```
## Content-Based Filtering

tfidf = TfidfVectorizer(stop_words='english') # Initialize TfidfVectorizer
tfidf_matrix = tfidf.fit_transform(books['Book-Title']) # Fit and transform book titles

nn = NearestNeighbors(metric='cosine', algorithm='brute') # Initialize NearestNeighbors
nn.fit(tfidf_matrix) # Fit the TF-IDF matrix

```

🔎 Collaborative Filtering

Pendekatan ini menggunakan metode Singular Value Decomposition (SVD) untuk mendekomposisi matriks pengguna-buku dan memprediksi rating.

```
## Collaborative Filtering (Matrix Factorization)
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
model = SVD()
cross_validate(model, data, cv=5, verbose=True)
```
## Evaluation
Untuk mengukur performa model rekomendasi, digunakan beberapa metrik evaluasi:

📊 Content-Based Filtering

Cosine Similarity Score: Mengukur kesamaan antara buku berdasarkan judul.

📊 Collaborative Filtering

Root Mean Squared Error (RMSE): Mengukur seberapa baik prediksi rating dibandingkan dengan rating asli.

Precision@K & Recall@K: Mengukur seberapa relevan rekomendasi yang dihasilkan.

```
# Evaluasi RMSE
trainset = data.build_full_trainset()
model.fit(trainset)
predictions = model.test(trainset.build_testset())
rmse = np.sqrt(mean_squared_error(ratings['Book-Rating'], [pred.est for pred in predictions]))
print("RMSE Score:", rmse)
```
```
RMSE Score: 0.9869750568359197
```

💡 Formula RMSE:

![image](https://github.com/user-attachments/assets/a05c80de-f628-420f-9d0b-f699982a97ae)

# Evaluasi Precision@K dan Recall@K

```
# Evaluasi Precision@K dan Recall@K
precision, recall = precision_recall_at_k(predictions, k=5, threshold=7)
print(f'Precision@5: {precision:.4f}')
print(f'Recall@5: {recall:.4f}')
```
```
Precision@5: 0.3378
Recall@5: 0.7647
```

Dengan evaluasi ini, kita dapat menilai apakah model sudah memberikan rekomendasi yang baik dan menentukan pendekatan mana yang lebih optimal.

💡 Visualisasi Evaluasi

![image](https://github.com/user-attachments/assets/23508af1-980f-4127-8aff-1678c2a94584)


# Kesimpulan

Dari proyek ini, dapat disimpulkan bahwa:

:star2: Content-Based Filtering efektif dalam merekomendasikan buku berdasarkan kesamaan fitur, tetapi terbatas pada buku yang memiliki informasi teks yang relevan.

:star2: Collaborative Filtering (SVD) memberikan rekomendasi yang lebih personal, tetapi membutuhkan data interaksi pengguna yang cukup untuk bekerja dengan baik.

:star2: RMSE digunakan sebagai metrik utama untuk Collaborative Filtering, sedangkan Content-Based Filtering menggunakan Cosine Similarity.

Proyek ini memberikan wawasan mengenai perbandingan dua metode rekomendasi dan dapat dikembangkan lebih lanjut dengan menggabungkan metode untuk meningkatkan akurasi.

