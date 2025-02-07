# Laporan Proyek Machine Learning - Salma Nurfauziah

## Domain Proyek

### Latar Belakang
Konsumsi makanan adalah sektor penting yang mendukung perekonomian Indonesia, namun rendahnya kualitas nutrisi dalam makanan yang dikonsumsi masyarakat Indonesia menjadi masalah besar dalam menjaga kesehatan. Berdasarkan data, masyarakat Indonesia lebih cenderung mengonsumsi makanan instan yang praktis dan cepat, namun dengan kandungan gizi yang seringkali tidak seimbang. Oleh karena itu, penting untuk melakukan analisis terhadap asupan gizi masyarakat Indonesia agar mereka bisa lebih memilih makanan yang sehat dan bergizi.

### Masalah yang Harus Diselesaikan:
Berdasarkan penelitian yang menunjukkan bahwa sekitar 68% populasi Indonesia tidak mampu memenuhi kebutuhan gizi harian yang seimbang, permasalahan ini perlu diselesaikan dengan memberikan pemahaman yang lebih baik tentang kandungan gizi makanan yang dikonsumsi masyarakat, serta membantu mereka dalam memilih makanan yang bergizi dan seimbang.

  Format Referensi: [Rendahnya Asupan Nutrisi Masyarakat Indonesia](https://www.kompas.id/baca/riset/2023/01/31/rendahnya-asupan-nutrisi-masyarakat-indonesia) 

## Business Understanding

### Problem Statements

Bagaimana cara mengidentifikasi dan menganalisis kandungan gizi dari makanan yang dikonsumsi oleh masyarakat Indonesia menggunakan dataset yang ada, dan memberikan rekomendasi makanan yang lebih bergizi berdasarkan analisis tersebut?

### Goals

- Mengembangkan sistem atau model yang dapat memberikan pemahaman tentang kandungan gizi dalam berbagai makanan yang dikonsumsi oleh masyarakat Indonesia.
- Menyediakan saran makanan yang bergizi untuk meningkatkan kualitas pola makan masyarakat.

    ### Solution statements
    - Pendekatan pertama: Menggunakan algoritma regresi untuk memprediksi kandungan gizi (seperti kalori, protein, lemak, karbohidrat) dari berbagai jenis makanan dalam dataset.
    - Pendekatan kedua: Membangun model klasifikasi yang mengategorikan makanan menjadi kategori sehat atau tidak sehat berdasarkan profil gizi mereka, dan memberikan rekomendasi berdasarkan hasil klasifikasi tersebut.

  Evaluasi dari solusi ini akan menggunakan metrik seperti Akurasi untuk klasifikasi dan RMSE (Root Mean Squared Error) untuk regresi, untuk memastikan bahwa model dapat memberikan rekomendasi yang relevan dan akurat.
      
## Data Understanding
Dataset ini terdiri dari informasi mengenai lebih dari seribu jenis makanan dan minuman khas Indonesia, dengan kolom-kolom seperti kalori, protein, lemak, karbohidrat, dan nama makanan. Contoh: [Kaggle - Indonesian Food and Drink Nutrition Dataset](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset) 

### Variabel-variabel pada Indonesian Food and Drink Nutrition Dataset adalah sebagai berikut:
- Calories: Kalori per porsi makanan.
- Proteins: Kandungan protein per porsi makanan.
- Fat: Kandungan lemak per porsi makanan.
- Carbohydrate: Kandungan karbohidrat per porsi makanan.
- Name: Nama makanan.
- Image: URL gambar makanan yang memberikan gambaran visual dari makanan tersebut.

**Exploratory Data Analysis (EDA)**:
- Sebelum melanjutkan ke tahap modeling, kita akan melakukan eksplorasi data untuk memahami distribusi kalori, protein, lemak, dan karbohidrat pada dataset ini. Visualisasi seperti histogram dan box plot akan membantu dalam mengeksplorasi pola dan potensi outlier dalam data.

  **Berikut adalah distribusi kalori, protein, lemak, dan karbohidrat pada dataset**:
  ![Distribusi Kalori, Protein, Lemak, dan Karbohidrat](https://drive.google.com/file/d/1pWWUn0e239nD3n1QCgL2p8DHM5i2dgI6/view?usp=drive_link)

## Data Preparation

**Teknik Data Preparation**: 
- Pembersihan Data: Menghapus data yang hilang atau tidak relevan, seperti entri dengan nilai kosong atau tidak valid.
- Normalisasi: Fitur-fitur seperti kalori, protein, lemak, dan karbohidrat perlu dinormalisasi agar berada dalam skala yang seragam untuk menghindari bias dalam model.
- Encoding: Jika diperlukan, kita akan melakukan encoding pada variabel kategorikal seperti nama makanan agar bisa digunakan dalam model.

**Alasan Data Preparation**:
Langkah-langkah ini penting untuk memastikan data siap digunakan dalam model machine learning. Pembersihan dan normalisasi data memastikan model bekerja dengan data yang akurat dan konsisten, serta tidak terpengaruh oleh masalah multikolinearitas atau skala yang berbeda pada fitur.

## Modeling

**Model yang Digunakan**: 
- Regresi Linier: Untuk memprediksi kandungan kalori, protein, lemak, atau karbohidrat berdasarkan fitur lainnya dalam dataset.
- Klasifikasi (Misalnya, KNN atau Decision Trees): Untuk mengkategorikan makanan menjadi kategori sehat atau tidak sehat berdasarkan profil gizi mereka.

**Kelebihan dan Kekurangan Algoritma**:
- Regresi Linier: Sederhana dan cepat diimplementasikan, namun terbatas dalam menangani hubungan non-linier antara fitur dan target.
- Klasifikasi: Misalnya, KNN memberikan hasil yang baik pada data yang tidak linear, tetapi bisa memerlukan waktu lebih lama untuk data besar.

Model akan dievaluasi berdasarkan metrik Akurasi untuk klasifikasi dan RMSE untuk regresi, dan jika diperlukan, model dapat diperbaiki dengan hyperparameter tuning untuk meningkatkan hasil.

## Evaluation

**Metrik Evaluasi yang Digunakan**:

- Regresi Linier: RMSE atau MSE untuk mengevaluasi seberapa baik model memprediksi kandungan gizi makanan.
- Klasifikasi: Akurasi, Precision, Recall, dan F1-Score untuk mengevaluasi kinerja model dalam mengklasifikasikan makanan sebagai sehat atau tidak sehat.
  
**Hasil Evaluasi**:
Model akan dievaluasi berdasarkan seberapa akurat prediksi atau klasifikasi yang dihasilkan, dengan fokus pada kemampuannya untuk memberikan saran makanan yang sehat dan bergizi sesuai dengan profil gizi yang dibutuhkan.

## Kesimpulan 
Laporan ini menunjukkan pentingnya analisis data untuk membantu masyarakat Indonesia dalam memilih makanan yang bergizi dan seimbang. Dengan menggunakan dataset Indonesian Food and Drink Nutrition Dataset, model ini bertujuan untuk memberikan rekomendasi yang dapat meningkatkan kualitas konsumsi gizi masyarakat, serta mengurangi prevalensi kekurangan gizi yang masih tinggi di Indonesia.


