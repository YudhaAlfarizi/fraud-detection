# 🔍 Fraud Detection: Deteksi Penipuan Transaksi Keuangan dengan Machine Learning

## 📌 Project Background
Proyek ini menganalisis dataset transaksi keuangan berukuran besar (**6.362.620 baris**) untuk mendeteksi transaksi penipuan. Penipuan hanya sekitar **0,13% (8.213 transaksi)** dari seluruh data, sehingga kelas data sangat tidak seimbang. Model yang dilatih dipakai di aplikasi web Streamlit untuk prediksi secara interaktif.

## 🧪 Alur Pengerjaan
1. **Setup:** import library dan memuat dataset
2. **EDA:** memahami struktur data, kelas target, jenis transaksi, jumlah, saldo, waktu, pelanggan, dan korelasi
3. **Persiapan Data:** memilih fitur, membagi data latih (70%) dan uji (30%) secara stratified
4. **Modeling:** pipeline `StandardScaler` + `OneHotEncoder` + Logistic Regression (`class_weight="balanced"`)
5. **Evaluasi:** classification report dan confusion matrix
6. **Deployment:** model disimpan dengan `joblib` dan dipakai di aplikasi Streamlit

## 📊 Temuan Kunci dan Insight
Setiap temuan dijelaskan artinya dan manfaatnya untuk deteksi penipuan.

| Temuan dari Data | Insight / Manfaat |
|---|---|
| Penipuan hanya 0,13% (8.213 dari 6.362.620 transaksi) | Akurasi menyesatkan: menebak semua transaksi "normal" saja sudah akurat sekitar 99,87%. Karena itu model dinilai dengan recall dan precision, dan memakai `class_weight="balanced"`. |
| Penipuan hanya muncul pada `TRANSFER` (sekitar 0,8%) dan `CASH_OUT` (sekitar 0,2%); jenis lain hampir nol | Pemantauan bisa diprioritaskan ke dua jenis transaksi ini, sehingga pemeriksaan lebih efisien (berlaku pada dataset ini). |
| Pada transaksi di bawah 50.000, median nominal penipuan terlihat kira-kira dua kali lipat transaksi normal | Nominal bisa menjadi salah satu sinyal peringatan. Namun sebaran kedua kelompok banyak yang tumpang tindih, jadi nominal tidak cukup dijadikan aturan tunggal. |
| Jumlah transaksi sangat miring (median 74.871, maksimum 92.445.516) | Nilai ekstrem perlu ditangani, sehingga fitur numerik diskalakan di dalam pipeline dan distribusi dilihat dengan skala logaritma. |
| Jumlah penipuan per `step` naik turun tanpa pola jelas | Waktu tidak memberi sinyal yang bisa dipakai, sehingga `step` dibuang dan model menjadi lebih sederhana. |
| Korelasi fitur dengan `isFraud` lemah (tertinggi `amount`, 0,08) | Tidak ada satu fitur pun yang cukup untuk mendeteksi penipuan, jadi dibutuhkan model yang menggabungkan beberapa fitur. |
| Saldo lama dan baru berkorelasi sangat kuat (pengirim 0,999; penerima 0,977) | Informasinya tumpang tindih. Perubahan saldo (selisih) lebih informatif untuk dikembangkan sebagai fitur baru. |
| 1.188.074 transaksi `TRANSFER`/`CASH_OUT` mengosongkan saldo pengirim, padahal total penipuan hanya 8.213 | Pola "saldo menjadi nol" terlalu umum. Kalau dijadikan aturan tunggal, sebagian besar yang ditandai justru bukan penipuan (banyak alarm palsu). |
| Pada 10 pengirim penipuan teratas, masing-masing hanya tercatat satu transaksi | Pelaku tidak terdeteksi lewat pengulangan akun, jadi deteksi perlu berbasis pola transaksi, bukan riwayat akun pengirim. |

## 🤖 Hasil Model

| Metrik (kelas penipuan) | Nilai |
|---|---|
| Recall | 0,95 |
| Precision | 0,02 |
| F1-score | 0,04 |
| Akurasi keseluruhan | sekitar 94,6% |

Pada data uji (1.908.786 transaksi, 2.464 di antaranya penipuan), model berhasil mendeteksi 2.341 penipuan dan melewatkan 123. Namun sekitar 100 ribu transaksi normal ikut ditandai sebagai penipuan, sehingga precision sangat rendah. Hasil bisa sedikit berbeda setiap dijalankan karena `train_test_split` tidak memakai `random_state`.

## 🗂️ Struktur Proyek
```
fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── Fraud_Detection_Analysis.ipynb   # EDA, preprocessing, pelatihan, dan evaluasi
├── app/
│   └── fraud_detection.py           # Aplikasi prediksi Streamlit
└── models/
    └── fraud_detection_pipeline.pkl # Pipeline hasil pelatihan
```

## 📁 Dataset
Dataset tidak disertakan di repository karena ukurannya besar.
1. Unduh dari [Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data)
2. Letakkan `AIML Dataset.csv` di folder yang sama dengan notebook

Dataset berisi sekitar 6,3 juta catatan transaksi, sebagian ditandai sebagai penipuan oleh sistem deteksi yang sudah ada. Jenis transaksi: `CASH-IN` (setor tunai ke merchant), `CASH-OUT` (tarik tunai), `DEBIT` (kirim ke rekening bank), `PAYMENT` (bayar barang/jasa), dan `TRANSFER` (kirim uang ke pengguna lain). Aplikasi Streamlit hanya mendukung `PAYMENT`, `TRANSFER`, dan `CASH_OUT`.

## 🚀 Cara Menjalankan
1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Buka `Fraud_Detection_Analysis.ipynb` di Jupyter Notebook, lalu jalankan semua sel untuk melihat analisis dan melatih model.
3. Jalankan aplikasi dengan Streamlit dari folder utama proyek:
   ```bash
   streamlit run app/fraud_detection.py
   ```

> Jika file `.pkl` gagal dimuat karena perbedaan versi scikit-learn, jalankan ulang notebook untuk membuat file model baru.

## 💡 Pengembangan Selanjutnya
* Mencoba model lain (misalnya Random Forest atau XGBoost)
* Memakai fitur selisih saldo yang sudah dibuat di notebook tetapi belum dipakai model
* Mengatur threshold prediksi untuk meningkatkan precision
* Mencoba teknik penanganan imbalance seperti SMOTE

## 🛠️ Teknologi
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Streamlit, joblib

## 🙏 Kredit
* **Referensi proyek:** mengikuti video YouTube *Fraud Detection Using Machine Learning – Full Python Data Science Project* oleh Onur Baltaci ([link video](ISI_LINK_VIDEO_DI_SINI)). Kode mengikuti video, dengan penjelasan diterjemahkan ke Bahasa Indonesia.
* **Dataset:** [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) oleh Aman Ali Siddiqui (Kaggle).
* **Tujuan:** pembelajaran dan portofolio, mendokumentasikan proses belajar saya dalam analisis data dan machine learning.

## 👤 Author
**[Nama Kamu]**
* [LinkedIn](ISI_LINK_LINKEDIN)
* [GitHub](https://github.com/<username-github-kamu>)
