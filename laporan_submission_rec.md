# Laporan Proyek Machine Learning - Salma Nurfauziah

## Project Overview

![Books](https://github.com/user-attachments/assets/44b105d4-843b-48b0-932e-8812eb447fc8)

Beragamnya jenis buku dan banyaknya jumlah buku merupakan permasalahan tersendiri bagi para pembaca buku. Salah satu permasalahan yang muncul adalah saat pembaca kesulitan dalam menentukan buku yang akan dibaca selanjutnya. Sistem rekomendasi buku sangat penting dalam era digital karena membantu pengguna menemukan buku yang sesuai dengan preferensi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi buku menggunakan **Content-Based Filtering, Collaborative Filtering, dan Neural Network-Based Recommender System** dengan dataset dari Kaggle.

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

Pada tahap ini, kita memahami dataset yang digunakan dalam proyek, termasuk jumlah data dan fitur yang ada.
Dataset terdiri dari:
- **Users**: Informasi pengguna seperti ID, lokasi, dan usia.
- **Ratings**: Data rating yang diberikan pengguna terhadap buku.
- **Books**: Informasi buku seperti judul, penulis, dan tahun publikasi.

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
- **Location**: Tempat tinggal pengguna berdasarkan informasi yang diberikan saat registrasi. Bisa berupa kota atau negara.
- **Age**: Usia pengguna dalam tahun, yang digunakan untuk memahami demografi pembaca.

🔍 Kondisi Data

1. Missing Values

* Users: Kolom `Age` memiliki nilai yang hilang.
* Books: Tidak ditemukan missing values.
* Ratings: Tidak ditemukan missing values.

2. Duplikat Data

* Books: Ditemukan beberapa ISBN yang memiliki entri ganda dengan informasi berbeda.
* Users & Ratings: Tidak ditemukan duplikasi pada User-ID dan kombinasi User-ID & ISBN.

3. Outlier

* Age: Terdapat nilai yang tidak realistis seperti usia 0 dan lebih dari 100 tahun.
* Book-Rating: Beberapa rating berada di luar rentang yang diharapkan (1-10).

Hasil analisis ini menunjukkan bahwa perlu dilakukan penanganan lebih lanjut pada tahap Data Preparation untuk mengatasi missing values, duplikasi, dan outlier.

🔍 Eksplorasi Data

✔ Memeriksa jumlah total pengguna, buku, dan rating.

✔ Menganalisis distribusi rating untuk melihat pola evaluasi buku oleh pengguna.

✔ Visualisasi hubungan antara pengguna dan buku berdasarkan rating.

:books: Books 

![book](https://github.com/user-attachments/assets/a83cfdc0-29bc-4026-84cd-b4608b19622f)

:chart_with_downwards_trend: Ratings

![rat](https://github.com/user-attachments/assets/a42bdce6-2f31-4718-bb5a-3b29cb9bd893)

:bust_in_silhouette: Users

![user](https://github.com/user-attachments/assets/6879cf64-39bf-41af-a1ee-5363e10238c6)

### Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami pola distribusi data, jumlah pengguna, buku, serta distribusi rating. 
- 10 penulis dengan jumlah buku terbanyak dianalisis, dengan Agatha Christie sebagai penulis teratas.

![penulis](https://github.com/user-attachments/assets/a4250ae2-516d-4d95-97a1-2cdf9fc2c21b)

- Distribusi rating menunjukkan mayoritas nilai berada dalam rentang tertentu.

:pushpin: Rating Buku

  ![rating](https://github.com/user-attachments/assets/dd76d610-f95b-489b-99df-b5758115b5aa)

  :pushpin: Rating per Pengguna
  
  ![pengguna](https://github.com/user-attachments/assets/ce8ca0ff-1bca-4d2f-aaf6-d44368b18ac3)

  Hasil EDA ini membantu dalam mempersiapkan data sebelum pemodelan. 🚀

## Data Preparation
**1. Tahap Preprocessing**

✔ Cleaning Data: Menghapus nilai yang hilang dan duplikasi.

✔ Menggabungkan Data: Menggabungkan data rating dengan book berdasarkan ISBN.

✔ Menangani Missing Values: Mengecek dan menghapus nilai yang hilang setelah penggabungan.

✔ Memfilter Data: Memilih hanya buku dengan jumlah rating cukup (minimal 5 review).

**2. Tahap Pembagian Data (train-test-split)**

✔ Persiapan Data untuk Collaborative Filtering: Menggunakan Surprise SVD untuk membangun model rekomendasi berbasis Collaborative Filtering .

**3. Tahap Ekstraksi Fitur**

✔ Ekstraksi Fitur dengan TF-IDF: Menggunakan TfidfVectorizer untuk mengubah data teks menjadi representasi numerik yang dapat digunakan untuk Content-Based Filtering.

**4. Hasil Akhir**

✔ Hasil Akhir Data: Menampilkan dataset yang sudah dibersihkan dan siap digunakan untuk pemodelan.

:pushpin: Tahap Preprocessing

1️⃣ Cleaning Data (Menghapus duplikasi & nilai yang hilang)
- Menghapus duplikasi pada dataset.
- Mengecek dan menghapus nilai yang hilang pada kolom Book-Author dan Publisher.

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
:exclamation: Terdapat missing value di bagian Book-Author dan Publisher.

![missing](https://github.com/user-attachments/assets/39a96833-92ae-4b13-a27d-ea3cc687914c)

2️⃣ Menggabungkan Data (Rating + Books)
- Menggabungkan data ratings dengan books berdasarkan ISBN agar setiap rating memiliki informasi buku yang sesuai.

```
# Menggabungkan dataframe ratings dengan books berdasarkan nilai ISBN
books = pd.merge(ratings, books, on='ISBN', how='left')
books
```
![gabung](https://github.com/user-attachments/assets/172dcad9-8b34-4e73-8789-7f8dbbe98cb2)

3️⃣ Menangani Missing Values Setelah Penggabungan
- Mengecek kembali apakah ada missing values setelah penggabungan.
- Menghapusnya jika diperlukan.
  
```
books.isnull().sum()  # Mengecek jumlah missing values
books.dropna(inplace=True)  # Menghapus baris dengan missing values jika perlu
# Menampilkan jumlah missing values di setiap kolom
print("Missing Values:\n", books.isnull().sum())
print("Data setelah menghapus missing values:", books.shape)
```

![value](https://github.com/user-attachments/assets/066054ec-9cbb-4df1-878b-45a11881ea25)

4️⃣ Memfilter Data dengan Rating Cukup
- Memastikan hanya buku yang memiliki minimal 5 rating yang digunakan agar data lebih valid.
  
```
book_counts = books.groupby('ISBN')['Book-Rating'].count()
books = books[books['ISBN'].isin(book_counts[book_counts >= 5].index)]
# Menghitung jumlah rating per ISBN

print("Jumlah data setelah memfilter buku dengan minimal 5 rating:", books.shape)

```
``
Jumlah data setelah memfilter buku dengan minimal 5 rating: (670480, 7)
``
:pushpin: Tahap Pembagian Data (train-test-split)

5️⃣ Persiapan Data untuk Collaborative Filtering

- Menggunakan Surprise SVD untuk membangun model rekomendasi berbasis Collaborative Filtering.
  
```
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Menyiapkan dataset untuk Surprise
reader = Reader(rating_scale=(0, 10))  # Sesuaikan dengan skala rating dataset-mu
data = Dataset.load_from_df(books[["User-ID", "ISBN", "Book-Rating"]], reader)

# Membagi data menjadi training & testing
trainset, testset = train_test_split(data, test_size=0.2)

# Menggunakan SVD dari Surprise
model = SVD(n_factors=20, random_state=42)
model.fit(trainset)

# Memprediksi pada test set
predictions = model.test(testset)

# Mengukur akurasi
rmse = accuracy.rmse(predictions)
print("SVD with Surprise RMSE:", rmse)

```
``
RMSE: 3.6694
SVD with Surprise RMSE: 3.669365879489593
``

:pushpin: Tahap Ekstraksi Fitur

6️⃣  Ekstraksi Fitur dengan TF-IDF

- Menggunakan TfidfVectorizer untuk mengubah data teks menjadi representasi numerik yang dapat digunakan untuk Content-Based Filtering.
  
```
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books["Book-Title"])

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
```
``
TF-IDF Matrix Shape: (670480, 21731)
``

✔ **Hasil Ekstraksi TF-IDF**  
* TF-IDF menghasilkan **21.731 fitur unik**, yang berarti ada 21.731 kata berbeda yang muncul dalam judul buku setelah stopwords dihapus. Ini menunjukkan bahwa dataset memiliki keragaman yang cukup tinggi dalam judul buku.

:pushpin: Hasil Akhir 

7️⃣ Hasil Akhir Data Setelah Dibersihkan
- Setelah semua tahapan data preparation dilakukan, dataset siap digunakan untuk pemodelan rekomendasi.
```
print("\nJumlah data setelah pembersihan:")
print(f"Books: {books.shape[0]} baris")
print(f"Users: {users.shape[0]} baris")
print(f"Ratings: {ratings.shape[0]} baris")
```

![hasil](https://github.com/user-attachments/assets/b75b4979-04b1-411b-a2c2-13c49a22e132)

:pushpin: Kesimpulan

✅ Tahapan data preparation berhasil membersihkan dataset dengan:

* Menghapus duplikasi.

* Menangani nilai yang hilang.

* Memastikan hanya buku dengan jumlah rating yang memadai digunakan dalam model.

📌 Manfaat:
Hal ini memastikan bahwa model rekomendasi akan bekerja dengan data yang lebih akurat dan relevan.
  
## **Modeling and Results**

📝 Pendekatan Model

Pada bagian ini digunakan dua pendekatan utama:

:sparkles: Content-Based Filtering → Mencari buku serupa berdasarkan fitur judul.

:sparkles: Collaborative Filtering (SVD) → Memprediksi rating berdasarkan pola rating pengguna lain.
 
📖 Content-Based Filtering
- Pendekatan ini menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk merepresentasikan fitur judul buku dalam bentuk vektor. Kemudian, algoritma Nearest Neighbors (k-NN) digunakan untuk menemukan buku serupa berdasarkan kemiripan vektor judul buku.
  
  ✔ Tahapan Model:
  * TF-IDF Vectorization → Mengubah teks judul buku menjadi representasi numerik.
  * k-Nearest Neighbors (k-NN) → Menggunakan metrik cosine similarity untuk mencari buku yang paling mirip.
```
## Content-Based Filtering

tfidf = TfidfVectorizer(stop_words='english') # Initialize TfidfVectorizer
tfidf_matrix = tfidf.fit_transform(books['Book-Title']) # Fit and transform book titles

nn = NearestNeighbors(metric='cosine', algorithm='brute') # Initialize NearestNeighbors
nn.fit(tfidf_matrix) # Fit the TF-IDF matrix

# Function to recommend books
def recommend_books_nn(title, n=5):
    if title not in books['Book-Title'].values:
        return f"Book '{title}' not found in the dataset."

    idx = books[books['Book-Title'] == title].index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)

    return books.iloc[indices[0][1:]][['Book-Title', 'Book-Author']]

# Example recommendation
recommend_books_nn("A Painted House")
```
![image](https://github.com/user-attachments/assets/6bf7da17-8858-4861-ab50-518311c88ae9)

🔹 Keunggulan: Tidak memerlukan data pengguna lain.

🔹 Kelemahan: Terbatas pada metadata yang tersedia.

🔎 Collaborative Filtering (Matrix Factorization - SVD)

- Pendekatan ini menggunakan Singular Value Decomposition (SVD) untuk mendekomposisi matriks pengguna-buku dan memprediksi rating buku berdasarkan pola rating pengguna lain.

  ✔ Tahapan Model:
  * Data Preprocessing → Menyiapkan dataset rating pengguna.
  * Matrix Factorization (SVD) → Mendekomposisi matriks pengguna-buku untuk menangkap pola laten.
  * Cross Validation → Mengukur performa model dengan RMSE dan MAE.
    
```
## Collaborative Filtering (Matrix Factorization)
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
model = SVD()
cross_validate(model, data, cv=5, verbose=True)
```

![image](https://github.com/user-attachments/assets/2c77c76c-da63-402a-9496-0fa5889d7ec5)

 ▶ **Hasil Rekomendasi Buku**
 - Top-N Rekomendasi Buku (Content-Based Filtering)
   
    | ISBN | Book-Title |	Book-Author |
    | ---- | ---------- | ----------- |
    | 80416 | Make Them Cry	| Kevin O'Brien |
    | 620850 | Make Them Cry | Kevin O'Brien |
    | 162070	| Make Them Cry	| Kevin O'Brien |
    | 15	| Make Them Cry	| Kevin O'Brien |
    | 355727	| Make Them Cry	| Kevin O'Brien |
   
  - Top-N Rekomendasi Buku (Collaborative Filtering - SVD)
  
  | Judul Buku | ISBN | Prediced Rating |
  | ------ | ----- | ------ |
  | Der Kleine Hobbit | 3423071516 | 10.00 |
  | Harry Potter und der Gefangene von Azkaban |  3551551693 | 10.00 |
  | Die Gefahrten I | 360893541X | 10.00 |
  | Die Wiederkehr Des Konigs III | 3608935436 | 10.00 |
  | Ender's Game (Ender Wiggins Saga (Paperback)) | 0812533550 | 10.00 |

🔹 Keunggulan: Mampu memberikan rekomendasi personal.

🔹 Kelemahan: Membutuhkan data interaksi yang cukup.

### **Kelebihan & Kekurangan**
| Pendekatan                | Kelebihan                                    | Kekurangan                                    |
|---------------------------|----------------------------------------------|----------------------------------------------|
| **Content-Based Filtering** | Tidak memerlukan data pengguna lain | Terbatas pada metadata yang tersedia |
| **Collaborative Filtering** | Mampu memberikan rekomendasi personal | Membutuhkan data interaksi yang cukup |

## Evaluation Model 
**1. Evaluasi Content-Based Filtering**
* Menggunakan Cosine Similarity untuk mengukur kesamaan antara buku berdasarkan metadata (judul, genre, deskripsi, dll.).
* Metrik evaluasi yang digunakan: Precision@K dan Recall@K untuk mengukur relevansi rekomendasi.

Formula Cosine Similarity:
![formula](https://github.com/user-attachments/assets/ec77fda9-1bb1-4ba8-af12-2084a70cdd28)

Hasil Evaluasi:

📌 Cosine Similarity menunjukkan bahwa buku dengan genre atau deskripsi yang mirip memiliki skor kemiripan yang lebih tinggi.

📌 Precision@K: 0.5, artinya 50% dari buku yang direkomendasikan memang relevan dengan preferensi pengguna.

📌 Recall@K: 0.5, menunjukkan bahwa 50% dari buku yang relevan berhasil ditemukan oleh sistem rekomendasi.

![cbf](https://github.com/user-attachments/assets/346525ac-5d69-415b-8d88-5210fad1b992)

Kesimpulan:

✔ Content-Based Filtering mampu merekomendasikan buku dengan karakteristik serupa berdasarkan metadata seperti genre dan deskripsi.

✔ Precision dan Recall bernilai sama (0.5), yang menunjukkan bahwa sistem cukup baik dalam menemukan buku yang relevan, tetapi masih ada ruang untuk perbaikan agar lebih optimal.

✔ Bisa ditingkatkan dengan hybrid approach yang mengombinasikan Content-Based dan Collaborative Filtering.

**2. Evaluasi Collaborative Filtering (SVD)**
- Evaluasi dilakukan menggunakan Root Mean Squared Error (RMSE) untuk mengukur error antara rating asli dan prediksi.

**Hasil Evaluasi:**

``
RMSE: 1.2406
RMSE Score (SVD): 1.2406265591732735
``

Hasil evaluasi RMSE: 1.2406, menunjukkan performa model cukup baik.

📌 Kesimpulan:

* SVD dengan RMSE 1.2406 menunjukkan performa yang cukup baik, tetapi masih ada ruang untuk perbaikan.
*  Model ini lebih unggul dalam memberikan rekomendasi personal dibanding Content-Based Filtering, karena mempertimbangkan pola interaksi pengguna secara lebih mendalam.
* Meskipun begitu, model masih memiliki error yang cukup signifikan, sehingga bisa dieksplorasi lebih lanjut dengan optimasi parameter atau metode hybrid.

**3. Evaluasi Neural Network-Based Recommender**
- Evaluasi dilakukan dengan melihat RMSE dari training dan validation loss.

📌 Analisis Hasil Evaluasi:
* Jika Train RMSE menurun tetapi Validation RMSE tetap tinggi, berarti model mengalami overfitting.
* Perlu dilakukan regularisasi tambahan atau tuning hyperparameter agar model tidak terlalu menyesuaikan data pelatihan.

✔ Solusi untuk Overfitting:
* Menambahkan dropout layer dalam arsitektur model.
* Menggunakan regularisasi L2 pada dense layer.
* Melakukan hyperparameter tuning pada jumlah neuron dan learning rate.
  
💡 Formula RMSE:

![rmse](https://github.com/user-attachments/assets/a05c80de-f628-420f-9d0b-f699982a97ae)

![hasil](https://github.com/user-attachments/assets/0916d84a-4789-47b6-b49c-3417e9ff886c)

💡 Visualisasi Evaluasi

Berdasarkan grafik RMSE:

✔ **Train RMSE menurun secara signifikan**, menunjukkan model belajar dengan baik pada data pelatihan.

✔ **Validation RMSE tetap tinggi dan cenderung stabil**, menunjukkan kemungkinan **overfitting**.

![visual](https://github.com/user-attachments/assets/78d4ae61-5ef1-4e46-9ba7-dbf00ec7fc68)

📊 Rekomendasi:
* Perlu dilakukan tuning hyperparameter lebih lanjut.
* Menggunakan teknik dropout dan regularisasi untuk meningkatkan generalisasi model.

# Kesimpulan

:star2: Content-Based Filtering cocok untuk rekomendasi berbasis metadata buku.

:star2: Collaborative Filtering lebih efektif dalam memberikan rekomendasi yang personal.

:star2: SVD menghasilkan RMSE yang lebih rendah, menunjukkan performa yang lebih stabil dibanding Neural Network.

:star2: Model Neural Network mengalami overfitting, sehingga perlu dilakukan perbaikan dengan metode regularisasi dan tuning hyperparameter.

Proyek ini memberikan wawasan mengenai perbandingan dua metode rekomendasi dan dapat dikembangkan lebih lanjut dengan menggabungkan metode untuk meningkatkan akurasi.

