import torch
from torch.utils.data import TensorDataset, DataLoader

from flux.utils import load_ae
from flux.models.text_encoder import load_qwen3_embedder
import pandas as pd
import os

from PIL import Image
from torchvision import transforms

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
        self.text_encoder.eval()
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
    def __init__(self, csv_data: str, output_dir: str = "images_output", batch_size: int = 16):
        self.csv_data = csv_data
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not os.path.exists(self.csv_data):
            raise ValueError(f"X CSV file not found: {self.csv_data}")

        print(f"Loading CSV from: {self.csv_data}")
        self.df = pd.read_csv(self.csv_data)
        print(f"CSV loaded: {len(self.df)} rows")

        print("Loading AutoEncoder...")
        self.ae = load_ae("./vae/ae.safetensors", device)
        self.ae.eval()
        print("AutoEncoder loaded")