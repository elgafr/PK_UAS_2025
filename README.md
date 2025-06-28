# Klasifikasi dan Retrieval Putusan Pengadilan Menggunakan Case-Based Reasoning
#
## Anggota Kelompok :
##
### Muhammad Hisyam Kamil - 202210370311060
### ELGA PUTRI TRI FARMA - 202210370311449
#

Repositori ini berisi kode dan data untuk penelitian yang bertujuan untuk mengembangkan sistem klasifikasi dan retrieval putusan pengadilan menggunakan pendekatan *Case-Based Reasoning* (CBR). Penelitian ini secara spesifik berfokus pada analisis kasus **pencucian uang**, dengan membandingkan performa dua metode representasi teks—**TF-IDF** (pendekatan statistik) dan **IndoBERT** (*deep learning* berbasis *text embeddings*)—serta mengimplementasikan model klasifikasi **Support Vector Machine (SVM)** dan sistem retrieval berbasis kemiripan kosinus.

## Deskripsi

Dalam ekosistem hukum modern, volume dokumen putusan pengadilan yang terus meningkat dan seringkali tidak terstruktur menimbulkan tantangan signifikan dalam pencarian preseden hukum yang relevan dan klasifikasi jenis perkara. Tantangan utama terletak pada representasi teks hukum yang padat terminologi spesifik ke dalam format yang dapat dipahami dan dianalisis oleh mesin secara efisien.

Penelitian ini mengimplementasikan kerangka kerja CBR untuk mengatasi permasalahan tersebut. Kami melakukan sistem klasifikasi otomatis dan retrieval cerdas dengan:
1.  **Mengakuisisi dan mempra-proses** 46 dokumen putusan dari Direktori Putusan Mahkamah Agung Republik Indonesia.
2.  **Mengekstrak fitur** menggunakan dua pipeline paralel: Representasi vektor TF-IDF dan *embeddings* kontekstual dari model IndoBERT.
3.  **Melatih klasifikator SVM** untuk mengkategorikan putusan ke dalam kelas 'Berat' atau 'Tidak Ditemukan' (berdasarkan kriteria masa pidana).
4.  **Mengimplementasikan sistem retrieval** berbasis kemiripan kosinus untuk menemukan dokumen paling relevan terhadap kueri.
5.  **Mengevaluasi kinerja** kedua model (TF-IDF dan IndoBERT) dalam klasifikasi dan retrieval secara komparatif.

Tujuan akhirnya adalah untuk menentukan pendekatan mana yang lebih andal dan akurat untuk domain hukum berbahasa Indonesia dengan volume data yang terbatas, serta mendemonstrasikan aplikasi prinsip CBR dalam konteks putusan pengadilan.

## Fitur Utama

-   **Akuisisi Data Otomatis:** Skrip untuk *web scraping* putusan pengadilan dari portal Mahkamah Agung dan penyimpanan data mentah (PDF dan teks) ke dalam CSV.
-   **Pra-pemrosesan dan Augmentasi Teks Hukum:**
    * Pembersihan dan standardisasi teks dari dokumen putusan.
    * Ekstraksi dan penambahan kolom metadata penting seperti `case_id`, `pihak` (penggugat vs. tergugat/pemohon vs. termohon), dan `ringkasan_fakta`.
    * Pelabelan otomatis (`label_klasifikasi`) putusan pidana ke dalam kategori 'Berat' atau 'Ringan' berdasarkan masa pidana yang tertera, serta kategori 'Tidak Ditemukan' untuk kasus non-pidana atau yang tidak teridentifikasi.
-   **Representasi Teks Multi-Metode:**
    * Representasi vektor menggunakan **TF-IDF** (dengan `max_features=5000`) untuk menangkap bobot kata kunci.
    * *Embeddings* kontekstual menggunakan model **IndoBERT** (768 dimensi) untuk menangkap makna semantik.
-   **Sistem Klasifikasi Otomatis:**
    * Melatih klasifikator **Support Vector Machine (SVM)** dengan *kernel linear* untuk setiap jenis representasi teks (`tfidf_matrix` dan `doc_embeddings_matrix`).
    * Penanganan kelas minoritas (kelas dengan kurang dari 2 sampel) secara otomatis untuk menghindari error stratifikasi pada `train_test_split`.
-   **Sistem Retrieval Informasi:**
    * Implementasi pencarian dokumen serupa menggunakan **kemiripan kosinus** antara kueri dan representasi dokumen (baik TF-IDF maupun IndoBERT embeddings).
    * Menampilkan hasil pencarian teratas (*top-N retrieval*) dengan skor kemiripan dan tautan dokumen.
-   **Demonstrasi *Solution Reuse*:** Menunjukkan bagaimana sistem dapat memprediksi solusi (label klasifikasi) untuk kueri baru berdasarkan kasus-kasus paling relevan yang ditemukan.
-   **Evaluasi Kinerja Komprehensif:**
    * Analisis perbandingan kinerja klasifikasi menggunakan metrik **Precision, Recall, F1-Score**, dan **Accuracy**, disajikan dalam bentuk laporan klasifikasi detail dan tabel ringkasan.
    * Visualisasi **Confusion Matrix** untuk analisis kesalahan klasifikasi.
    * Evaluasi kinerja retrieval menggunakan metrik **Hit@5**.
    * Analisis kesalahan prediksi untuk mengidentifikasi kasus-kasus yang salah diklasifikasikan.

## Memulai

### 1. Prasyarat

-   Python 3.8 atau versi lebih baru
-   `pip` (manajer paket Python)
-   `virtualenv` (direkomendasikan untuk manajemen lingkungan)

### 2. Instalasi

Untuk menyiapkan lingkungan dan menginstal semua dependensi yang diperlukan, ikuti langkah-langkah berikut:

1.  **Klon Repositori**
    ```bash
    git clone [https://github.com/hisyam99/PK_UAS_2025.git](https://github.com/hisyam99/PK_UAS_2025.git)
    cd PK_UAS_2025
    ```

2.  **Buat dan Aktifkan Lingkungan Virtual (Direkomendasikan)**
    ```bash
    # Untuk macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Untuk Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instal Dependensi**
    Instal semua pustaka Python yang dibutuhkan dari file `requirements.txt`. Pastikan file `requirements.txt` mencakup semua pustaka yang digunakan dalam proyek ini, seperti `pandas`, `scikit-learn`, `transformers`, `torch`, `sentencepiece`, `numpy`, `matplotlib`, `seaborn`, `requests`, `beautifulsoup4`, `pdfplumber`, dll.
    ```bash
    pip install -r requirements.txt
    ```

### 3. Struktur Direktori Proyek

Pastikan struktur direktori proyek Anda kira-kira seperti ini agar semua path bekerja dengan benar:

````

Case-Based-Reasoning-Penalaran-Komputer/
├── notebooks/
│   └── case-based-reasoning.ipynb  \# Notebook utama proyek
├── data/
│   ├── raw/                      \# Untuk data mentah hasil scraping (opsional)
│   ├── processed/                \# Untuk data CSV yang sudah diproses (processed\_cases.csv)
│   │   └── processed\_cases.csv
│   └── eval/                     \# Untuk menyimpan queries.json, retrieval\_metrics.csv, prediction\_metrics.csv
├── CSV/                          \# Untuk menyimpan CSV hasil scraping sementara (putusan\_ma\_url\_scrape\_2025-06-28.csv)
├── RAW\_TEXT/                     \# Untuk menyimpan teks mentah dari PDF (case\_XXX.txt)
├── README.md
└── requirements.txt

## ⚙ Cara Menjalankan Pipeline End-to-End

Seluruh pipeline penelitian, mulai dari akuisisi data, pra-pemrosesan, representasi, pelatihan model, hingga evaluasi, dijalankan melalui notebook Jupyter.

### Menjalankan Notebook

Cara utama untuk menjalankan pipeline ini adalah dengan membuka dan menjalankan sel-sel kode di dalam notebook Jupyter utama Anda.

1.  **Luncurkan Jupyter Notebook**
    Dari direktori utama proyek, jalankan perintah berikut di terminal Anda:
    ```bash
    jupyter notebook
    ```

2.  **Buka dan Jalankan Notebook**
    * Di browser Anda, navigasikan ke direktori tempat notebook Anda berada (misalnya `notebooks/`).
    * Buka file `case-based-reasoning.ipynb` (atau nama file notebook utama Anda).
    * **Penting:** Jalankan setiap sel kode secara berurutan (dari atas ke bawah) untuk mereplikasi seluruh alur kerja. Pastikan untuk mengizinkan akses ke Google Drive jika Anda menggunakan Google Colab, karena beberapa file disimpan di sana.

### Contoh Perintah

Untuk memulai, pastikan Anda berada di direktori utama repositori dan lingkungan virtual Anda aktif. Kemudian jalankan:

```bash
jupyter notebook notebooks/case-based-reasoning.ipynb
````
