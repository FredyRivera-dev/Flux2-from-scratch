from dotenv import load_dotenv
import os
from huggingface_hub import hf_hub_download
import requests
import uuid

load_dotenv()

dataset = os.getenv("DATASET", "")
file_data = os.getenv("FILEDATA", "")

def download_dataset_file(output_dir="./data_file"):
    hf_hub_download(repo_id=dataset, filename=file_data, repo_type="dataset", local_dir=output_dir)

    return f"{output_dir}/{file_data}"

def temp_download_image(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        save_path = f"{uuid.uuid4().hex}.png"

        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully to {save_path}")

        return save_path

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

## temp_download_image("https://f004.backblazeb2.com/file/sota-images/557adae6-1a3c-4f76-a4b5-2bc954d65cce.png")