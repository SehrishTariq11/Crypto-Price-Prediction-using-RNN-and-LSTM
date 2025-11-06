import os
import gdown
import zipfile

def setup_models():
    """
    Checks if models_output exists. 
    If not, downloads models_output.zip from Google Drive and extracts it.
    """
    models_dir = "models_output"
    zip_file = "models_output.zip"

    # ✅ If already extracted, skip download
    if os.path.exists(models_dir):
        print("✅ Models folder already exists — skipping download.")
        return

    # ⚙️ Google Drive ZIP file ID (replace with your file's ID)
    FILE_ID = "https://drive.google.com/file/d/1Hz4UFpIbNdwhSJZlLUIxCIP4OMxnzKWJ/view?usp=sharing"

    # Build Google Drive direct download link
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("⬇️ Downloading models_output.zip from Google Drive...")
    gdown.download(url, zip_file, quiet=False)

    # Extract ZIP file
    print("📦 Extracting models_output.zip ...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print("✅ Extraction complete! Models ready to use.")

# -----------------------------
# Call setup before running Streamlit app
# -----------------------------
setup_models()
