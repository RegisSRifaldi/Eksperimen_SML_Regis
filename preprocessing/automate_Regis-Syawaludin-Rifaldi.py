
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Lokasi dataset dan folder penyimpanan
RAW_DATA_PATH = "../data/raw/bank_full_raw.csv"
PROCESSED_DIR = "processed/"


def load_data():
    # Membaca dataset CSV
    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    print("Dataset berhasil dimuat.")
    print(f"Jumlah baris: {df.shape[0]}")
    print(f"Jumlah kolom: {df.shape[1]}")

    return df


def preprocess_data(df):
    # 1. Menghapus kolom duration untuk menghindari data leakage
    df = df.drop(columns=["duration"])

    # 2. Memisahkan fitur dan target
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    # 3. Memisahkan fitur numerik dan kategorikal
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\n=== INFORMASI FITUR ===")
    print(f"Jumlah fitur: {X.shape[1]}")
    print(f"Jumlah fitur numerik: {len(numeric_features)}")
    print(f"Jumlah fitur kategorikal: {len(categorical_features)}")

    # 4. Membagi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Membuat proses preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )

    # 6. Melakukan preprocessing pada data training
    X_train_processed = preprocessor.fit_transform(X_train)

    # 7. Melakukan preprocessing pada data testing
    X_test_processed = preprocessor.transform(X_test)

    # 8. Mengambil nama fitur hasil preprocessing
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    print("\n=== HASIL PREPROCESSING ===")
    print(f"Data training: {X_train_processed.shape}")
    print(f"Data testing: {X_test_processed.shape}")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    # Membuat folder processed jika belum tersedia
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Menyimpan data training dan testing
    X_train.to_csv(PROCESSED_DIR + "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR + "X_test.csv", index=False)

    # Menyimpan target training dan testing
    y_train.to_csv(
        PROCESSED_DIR + "y_train.csv",
        index=False,
        header=["y"]
    )

    y_test.to_csv(
        PROCESSED_DIR + "y_test.csv",
        index=False,
        header=["y"]
    )

    # Menyimpan proses preprocessing
    joblib.dump(
        preprocessor,
        PROCESSED_DIR + "preprocessor.joblib"
    )

    print("\nSemua hasil preprocessing berhasil disimpan.")

def main():
    # Membaca dataset
    df = load_data()

    # Melakukan preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Menyimpan hasil preprocessing
    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

if __name__ == "__main__":
    main()
