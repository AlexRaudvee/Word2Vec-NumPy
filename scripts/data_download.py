import os
import requests
import zipfile

from io import BytesIO

from datasets import load_dataset

def download_and_extract(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(target_dir)
    print(f"Downloaded and extracted to {target_dir}")

def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded file to {target_path}")

download_and_extract(
    "https://staff.fnwi.uva.nl/e.bruni/resources/MEN.zip",
    "data"
)

download_file(
    "https://www.dropbox.com/s/ne0fib302jqbatw/EN-MSR.txt?dl=1",
    "data/msr/msr.txt"
)

ag_news = load_dataset("fancyzhx/ag_news")
ag_news_dir = "data/ag_news"
os.makedirs(ag_news_dir, exist_ok=True)
ag_news.save_to_disk(ag_news_dir)
print(f"ag_news dataset downloaded and saved to {ag_news_dir}")

wikitext2_dir = "data/wikitext-2"
os.makedirs(wikitext2_dir, exist_ok=True)
wikitext2_files = ["train.txt", "valid.txt", "test.txt"]
wikitext2_base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"

if os.path.isfile(os.path.join(wikitext2_dir, "train.txt")):
    print("WikiText-2 already downloaded.")
else:
    print("Downloading WikiText-2 (PyTorch mirror)...")
    for fname in wikitext2_files:
        url = f"{wikitext2_base_url}/{fname}"
        local_path = os.path.join(wikitext2_dir, fname)
        print(f"Downloading {url} to {local_path}")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
    print("Download complete.")

for fname in wikitext2_files:
    local_path = os.path.join(wikitext2_dir, fname)
    if os.path.isfile(local_path):
        print(f"{fname}: {os.path.getsize(local_path)} bytes")