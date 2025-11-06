import os
import zipfile
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
dataset_zip = "preprocessed_datasets.zip"
dataset_folder = "preprocessed_datasets"

models_zip = "trained_models.zip"
models_folder = "trained_models"

# -----------------------------
# Function to ensure folder exists
# -----------------------------
def ensure_folder(folder_path, zip_path):
    if not os.path.exists(folder_path):
        if os.path.exists(zip_path):
            print(f"📦 Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
            print(f"✅ Extracted to {folder_path}")
        else:
            raise FileNotFoundError(
                f"❌ Neither {folder_path} folder nor {zip_path} found! "
                "Please provide the required files."
            )

# -----------------------------
# Ensure dataset and models folders exist
# -----------------------------
ensure_folder(dataset_folder, dataset_zip)
ensure_folder(models_folder, models_zip)

# -----------------------------
# Load dataset CSV files
# -----------------------------
data_files = [
    os.path.join(dataset_folder, f)
    for f in os.listdir(dataset_folder)
    if f.endswith(".csv")
]

if not data_files:
    raise FileNotFoundError(f"❌ No CSV files found in {dataset_folder}!")

for file in data_files:
    df = pd.read_csv(file)
    print(f"\nLoaded {file} (first 2 rows):")
    print(df.head(2))

# -----------------------------
# Load trained models
# -----------------------------
model_files = [
    os.path.join(models_folder, f)
    for f in os.listdir(models_folder)
    if f.endswith(".h5") or f.endswith(".pkl")  # adjust to your model format
]

if not model_files:
    raise FileNotFoundError(f"❌ No model files found in {models_folder}!")

print(f"\n✅ Found {len(model_files)} trained models in {models_folder}")
