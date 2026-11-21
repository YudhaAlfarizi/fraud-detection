<div align="center">
    
# 🔍 Fraud Detection: Deteksi Penipuan Transaksi Keuangan dengan Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LogisticRegression-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerik-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisasi-11557c)

</div>

## 📌 Project Background
Dataset yang dipakai berisi **6.362.620 catatan transaksi keuangan** dengan 11 kolom, yaitu:

| Kolom | Keterangan |
|---|---|
| `step` | Satuan waktu terjadinya transaksi |
| `type` | Jenis transaksi (`CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`) |
| `amount` | Jumlah uang yang ditransaksikan |
| `nameOrig` | Identitas pelanggan pengirim |
| `oldbalanceOrg` | Saldo pengirim sebelum transaksi |
| `newbalanceOrig` | Saldo pengirim setelah transaksi |
| `nameDest` | Identitas penerima |
| `oldbalanceDest` | Saldo penerima sebelum transaksi |
| `newbalanceDest` | Saldo penerima setelah transaksi |
| `isFraud` | Label target: 1 = penipuan, 0 = transaksi normal |
| `isFlaggedFraud` | Penanda transaksi yang ditandai otomatis oleh sistem deteksi yang sudah ada |

Pada transaksi keuangan digital, penipuan sulit dikenali karena jumlahnya sangat sedikit dibanding transaksi normal yang sangat banyak. Proyek ini bertujuan untuk **memahami karakteristik transaksi penipuan melalui analisis data**, lalu **membangun model machine learning yang dapat mengklasifikasikan transaksi sebagai penipuan atau bukan**. Model yang dilatih dipakai di aplikasi web Streamlit untuk prediksi secara interaktif.

## 🧪 Alur Pengerjaan
1. **Setup:** import library dan memuat dataset
2. **EDA:** memahami struktur data, kelas target, jenis transaksi, jumlah, saldo, waktu, pelanggan, dan korelasi
3. **Persiapan Data:** memilih fitur, membagi data latih (70%) dan uji (30%) secara stratified
4. **Modeling:** pipeline `StandardScaler` + `OneHotEncoder` + Logistic Regression (`class_weight="balanced"`)
5. **Evaluasi:** classification report dan confusion matrix
6. **Deployment:** model disimpan dengan `joblib` dan dipakai di aplikasi Streamlit

## 💡 Key Insights
* **Data sangat tidak seimbang (penipuan 0,13%):** akurasi menyesatkan, jadi model dinilai dengan recall dan precision.
* **Penipuan hanya muncul pada `TRANSFER` dan `CASH_OUT`:** dua jenis ini bisa diprioritaskan dalam pemantauan.
* **Nominal penipuan cenderung lebih besar, tetapi banyak tumpang tindih dengan transaksi normal:** nominal tidak cukup dijadikan aturan tunggal.
* **Korelasi tiap fitur dengan penipuan lemah, dan saldo lama–baru saling tumpang tindih:** dibutuhkan kombinasi fitur, dan selisih saldo layak dikembangkan.
* **1.188.074 transaksi mengosongkan saldo pengirim, padahal penipuan hanya 8.213:** pola ini saja akan menghasilkan banyak alarm palsu.

## 🏆 Achievements
* Menganalisis 6,3 juta transaksi keuangan untuk mengidentifikasi pola penipuan yang terkonsentrasi pada transaksi `TRANSFER` dan `CASH_OUT`.
* Membangun pipeline Logistic Regression yang mendeteksi sekitar 95% transaksi penipuan pada data uji (recall 0,95), dengan precision 0,02 sehingga masih banyak alarm palsu.
* Membuat aplikasi web Streamlit untuk memprediksi transaksi penipuan secara interaktif.

> Hasil bisa sedikit berbeda setiap dijalankan karena `train_test_split` tidak memakai `random_state`.

## Struktur Proyek
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

## 🔭 Pengembangan Selanjutnya
* Mencoba model lain (misalnya Random Forest atau XGBoost)
* Memakai fitur selisih saldo yang sudah dibuat di notebook tetapi belum dipakai model
* Mengatur threshold prediksi untuk meningkatkan precision
* Mencoba teknik penanganan imbalance seperti SMOTE

## 🛠️ Teknologi
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Streamlit, joblib

## 🙏 Kredit
* **Referensi proyek:** mengikuti video YouTube *Fraud Detection Using Machine Learning – Full Python Data Science Project* oleh Onur Baltaci ([link video](https://www.youtube.com/watch?v=4Od5_z28iIE)). Kode mengikuti video, dengan penjelasan diterjemahkan ke Bahasa Indonesia.
* **Dataset:** [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) oleh Aman Ali Siddiqui (Kaggle).

## 👤 Author
**YUDHA ALFARIZI**
* [LinkedIn](https://www.linkedin.com/in/yudha-alfarizi-07588a286/)
