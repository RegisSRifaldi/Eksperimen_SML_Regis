# Eksperimen Machine Learning — Bank Marketing

## Deskripsi Proyek

Proyek ini merupakan implementasi eksperimen Machine Learning menggunakan Bank Marketing Dataset dari UCI Machine Learning Repository.

Tujuan eksperimen adalah mempersiapkan dataset untuk membangun model klasifikasi biner yang memprediksi apakah nasabah akan berlangganan deposito berjangka (_term deposit_).

## Dataset

- **Nama:** Bank Marketing Dataset
- **Sumber:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Jumlah data:** 45.211 observasi
- **Jumlah kolom:** 17
- **Target:** `y` (yes/no)
- **Jenis masalah:** Binary Classification

## Struktur Proyek

```text
Eksperimen_SML_X/
├── data/
│   └── raw/
│       └── bank_full_raw.csv
├── preprocessing/
│   ├── processed/
│   ├── aautomate_Regis-Syawaludin-Rifaldi.py
│   └── Eksperimen_Regis-Syawaludin-Rifaldi.ipynb
├── requirements.txt
└── README.md
```

## Tahapan Eksperimen

1. Pengenalan dataset.
2. Import library.
3. Memuat dataset.
4. Exploratory Data Analysis (EDA).
5. Data preprocessing.
6. Otomatisasi preprocessing.

## Preprocessing

Tahapan preprocessing meliputi:

- Menghapus kolom `duration` untuk menghindari data leakage.
- Mengubah target `y` menjadi nilai biner (no = 0, yes = 1).
- Membagi dataset menjadi data training dan testing dengan rasio 80:20.
- Melakukan standardisasi fitur numerik menggunakan StandardScaler.
- Melakukan encoding fitur kategorikal menggunakan OneHotEncoder.

## Hasil Preprocessing

| Keterangan                 |  Hasil |
| -------------------------- | -----: |
| Data awal                  | 45.211 |
| Fitur sebelum transformasi |     15 |
| Fitur setelah transformasi |     50 |
| Data training              | 36.168 |
| Data testing               |  9.043 |

## Cara Menjalankan

Instal dependensi:

```bash
pip install -r requirements.txt
```

Jalankan otomatisasi preprocessing dari root repository:

```bash
python preprocessing/automate_Regis-Syawaludin-Rifaldi.py
```

## Teknologi

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
