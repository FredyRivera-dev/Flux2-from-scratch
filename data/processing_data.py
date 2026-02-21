import torch

from flux.utils import load_ae
from flux.models.text_encoder import load_qwen3_embedder
import pandas as pd
import os

from PIL import Image
from torchvision import transforms
from pathlib import Path

device = "cuda:0"

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize to expected resolution
    transforms.ToTensor(),           # Convert to tensor [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

class TextEncoderProcess:
    def __init__(self, csv_data: str, output_dir: str = "embeddings_output", batch_size: int = 16):
        self.csv_data = csv_data
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not os.path.exists(self.csv_data):
            raise ValueError(f"X CSV file not found: {self.csv_data}")

        print(f"Loading CSV from: {self.csv_data}")
        self.df = pd.read_csv(self.csv_data)
        print(f"CSV loaded: {len(self.df)} rows")

        if 'prompt' not in self.df.columns:
            raise ValueError("X CSV must have a 'prompt' column")

        print("Loading text encoder...")
        self.text_encoder = load_qwen3_embedder(variant="4B", device=device)
        print("Text encoder loaded")
    
    def process(self):
        total_samples = len(self.df)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        print(f"\nStarting processing:")
        print(f"  Total samples: {total_samples}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total batches: {num_batches}")
        print(f"  Output dir: {self.output_dir}\n")
        
        sample_idx = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)

            batch_df = self.df.iloc[start_idx:end_idx]
            prompts = batch_df['prompt'].tolist()
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx-1})...")
            
            try:
                with torch.no_grad():
                    emb_batch = self.text_encoder(prompts).to(torch.bfloat16)

                for i, (emb, prompt_str) in enumerate(zip(emb_batch, prompts)):
                    data = {
                        "emb": emb.cpu().to(torch.bfloat16),
                        "prompt_str": prompt_str,
                        "embedding_shape": list(emb.shape),
                        "csv_index": start_idx + i,
                    }

                    filename = f"sample_{sample_idx:06d}.pt"
                    filepath = self.output_dir / filename
                    torch.save(data, filepath)
                    
                    sample_idx += 1
                
                print(f"Saved {len(prompts)} embeddings")
                
            except Exception as e:
                print(f"  X Error in batch {batch_idx + 1}: {type(e).__name__}: {e}")
                continue
            
            finally:
                del emb_batch
                torch.cuda.empty_cache()
        
        print(f"\nComplete processing: {sample_idx}/{total_samples} samples saved")
        return sample_idx

class ImageProcess:
    def __init__(self, csv_data: str, images_dir: str, output_dir: str = "images_output", batch_size: int = 16):
        self.csv_data = csv_data
        self.batch_size = batch_size
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not os.path.exists(self.csv_data):
            raise ValueError(f"X CSV file not found: {self.csv_data}")

        if not self.images_dir.exists():
            raise ValueError(f"X Images directory not found: {self.images_dir}")

        print(f"Loading CSV from: {self.csv_data}")
        self.df = pd.read_csv(self.csv_data)
        print(f"CSV loaded: {len(self.df)} rows")

        self.image_cols = [col for col in self.df.columns if col.startswith('image_')]
        if not self.image_cols:
            raise ValueError("X CSV must have at least one 'image_*' column")
        print(f"Found {len(self.image_cols)} image columns: {self.image_cols}")

        print("Loading AutoEncoder...")
        self.ae = load_ae("./vae/ae.safetensors", device)
        self.ae.eval()
        print("AutoEncoder loaded")


    def load_image(self, image_name: str) -> torch.Tensor:
        image_path = self.images_dir / image_name
        
        if not image_path.exists():
            raise FileNotFoundError(f"X Image not found: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        return img_tensor

    def process(self):
        all_images_data = []

        for idx, row in self.df.iterrows():
            for img_col in self.image_cols:
                image_name = row[img_col]
                if pd.notna(image_name) and image_name:
                    all_images_data.append({
                        'image_name': image_name,
                        'csv_index': idx,
                        'column': img_col
                    })

        total_images = len(all_images_data)
        num_batches = (total_images + self.batch_size - 1) // self.batch_size

        print(f"\nStarting image processing:")
        print(f"  Total images: {total_images}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total batches: {num_batches}")
        print(f"  Output dir: {self.output_dir}\n")
        
        latent_idx = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_images)
            
            batch_images_data = all_images_data[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (images {start_idx}-{end_idx-1})...")
            
            try:
                images_batch = []
                valid_data = []
                
                for img_data in batch_images_data:
                    try:
                        img_tensor = self.load_image(img_data['image_name'])
                        images_batch.append(img_tensor)
                        valid_data.append(img_data)
                    except Exception as e:
                        print(f"  X Error loading {img_data['image_name']}: {e}")
                        continue
                
                if not images_batch:
                    print(f"  X No valid images in batch {batch_idx + 1}")
                    continue

                batch = torch.stack(images_batch).to(device)

                with torch.no_grad():
                    latents_batch = self.ae.encode(batch).to(torch.bfloat16)

                for latent, img_data in zip(latents_batch, valid_data):
                    data = {
                        "latents": latent.cpu().to(torch.bfloat16),
                        "image_name": img_data['image_name'],
                        "latent_shape": list(latent.shape),
                        "csv_index": img_data['csv_index'],
                        "csv_column": img_data['column']
                    }
                    
                    filename = f"latent_{latent_idx:06d}.pt"
                    filepath = self.output_dir / filename
                    torch.save(data, filepath)
                    
                    latent_idx += 1
                
                print(f"Saved {len(valid_data)} latents")
                
            except Exception as e:
                print(f"X Error in batch {batch_idx + 1}: {type(e).__name__}: {e}")
                continue
            
            finally:
                if 'batch' in locals():
                    del batch
                if 'latents_batch' in locals():
                    del latents_batch
                torch.cuda.empty_cache()
        
        print(f"\nComplete processing: {latent_idx}/{total_images} latents saved")
        return latent_idx

class UnifyData:
    def __init__(self, embeddings_pt_dir: str, latents_pt_dir: str):
        self.embeddings_pt_dir = embeddings_pt_dir
        self.latents_pt_dir = latents_pt_dir

if __name__ == "__main__":

    text_processor = TextEncoderProcess(
        csv_data="./data/data.csv",
        batch_size=16
    )

    text_processor.process()

    image_processor = ImageProcess(
        csv_data="./data/data.csv",
        images_dir="./images",
        batch_size=8
    )

    image_processor.process()