# Okay, in this file I'm going to leave an example 
# of how to prepare a dataset. The resulting 
# dataset will be made public on HuggingFace: https://huggingface.co/datasets/Fredtt3/Flux2-Image

import pandas as pd
from .filter import contains_child_reference, child_related_terms
from .utils import download_dataset_file, temp_download_image
from .grid_image import split_grid_image
import os

class PrepareDataFlux2:
    def __init__(self, csv_data: str):
        self.csv_data = csv_data

        if not os.path.exists(self.csv_data):
            target_dir = os.path.dirname(self.csv_data)

            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)

            download_dataset_file(target_dir)

    def prepare(self, output_csv: str, image_dir: str, prompt_field: str = "PROMPT", image_field: str = "IMAGE_URL", download_image: bool = True):
        pass

    def filter_data(self, prompt: str):
        return contains_child_reference(prompt, child_related_terms)