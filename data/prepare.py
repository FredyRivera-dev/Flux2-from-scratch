# Okay, in this file I'm going to leave an example 
# of how to prepare a dataset. The resulting 
# dataset will be made public on HuggingFace: https://huggingface.co/datasets/Fredtt3/Flux2-Image

import pandas as pd
from .filter import contains_child_reference, child_related_terms
from .utils import download_dataset_file, temp_download_image, file_data
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

    def prepare(self, output_csv: str, image_dir: str, 
                prompt_field: str = "PROMPT", image_field: str = "IMAGE_URL", 
                download_image: bool = True, batch_size: int = 1000, max_samples: int = None):

        df = pd.read_csv(self.csv_data)

        if max_samples is not None:
            df = df.head(max_samples)
    
        print(f"Total number of records: {len(df)}")

        print("Filtering inappropriate content...")
        df['is_safe'] = df[prompt_field].apply(lambda x: not self.filter_data(str(x)))
        df_safe = df[df['is_safe']].copy()
        print(f"Safe records after filtering: {len(df_safe)}")

        os.makedirs(image_dir, exist_ok=True)

        results = []

        total_batches = (len(df_safe) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            print(f"\n--- Processing Batch {batch_num + 1}/{total_batches} ---")

            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(df_safe))
            batch_df = df_safe.iloc[start_idx:end_idx].copy()
        
            print(f"Processing {len(batch_df)} images in this batch...")
        
            temp_images = []
        
            for idx, row in batch_df.iterrows():
                try:
                    prompt = row[prompt_field]
                    image_url = row[image_field]

                    if download_image:
                        temp_image_path = temp_download_image(image_url)
                        if temp_image_path is None:
                            print(f"Error downloading image {image_url}")
                            continue
                    
                        temp_images.append(temp_image_path)
                    
                        grid_images = split_grid_image(
                            temp_image_path, 
                            output_dir=image_dir,
                            rows=2, 
                            cols=2,
                            return_paths=False
                        )
                    
                        result_row = {'prompt': prompt}
                        for i, grid_img_name in enumerate(grid_images, 1):
                            result_row[f'image_{i}'] = grid_img_name
                    
                        results.append(result_row)
                
                except Exception as e:
                    print(f"Error processing record {idx}: {str(e)}")
                    continue

            print(f"Cleaning {len(temp_images)} temporary images...")
            for temp_img in temp_images:
                try:
                    if os.path.exists(temp_img):
                        os.remove(temp_img)
                except Exception as e:
                    print(f"Error removing {temp_img}: {str(e)}")
        
            results_df = pd.DataFrame(results)
            if batch_num == 0:
                results_df.to_csv(output_csv, index=False, mode='w')
            else:
                results_df.to_csv(output_csv, index=False, mode='a', header=False)
        
            print(f"Batch {batch_num + 1} completed. Total rows generated: {len(results)}")
            results = []
    
        print(f"\nProcess completed. Results saved in {output_csv}")
        print(f"Images saved in {image_dir}")

    def filter_data(self, prompt: str):
        return contains_child_reference(prompt, child_related_terms)

if __name__ == "__main__":
    data_prepare = PrepareDataFlux2(f"./data/{file_data}")

    ## Only test
    data_prepare.prepare(output_csv="./output/data.csv", image_dir="./output/images",
                        batch_size=100, max_samples=200)