import torch
from torch.utils.data import TensorDataset, DataLoader

from flux.utils import load_ae
from flux.models.text_encoder import load_qwen3_embedder

# device = "cuda:0"

# ae = load_ae("./vae/ae.safetensors", device)
# ae.eval()

# text_encoder = load_qwen3_embedder(variant="4B", device=device)
# text_encoder.eval()