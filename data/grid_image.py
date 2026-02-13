from PIL import Image
import os
import sys
import uuid

def split_grid_image(input_path, output_dir="output", rows=2, cols=2, return_paths=False):
    names = []
    try:

        img = Image.open(input_path)
        width, height = img.size

        cell_width = width // cols
        cell_height = height // rows

        os.makedirs(output_dir, exist_ok=True)

        for row in range(rows):
            for col in range(cols):
                left = col * cell_width
                top = row * cell_height
                right = left + cell_width
                bottom = top + cell_height

                cropped = img.crop((left, top, right, bottom))

                output_filename = f"{uuid.uuid4().hex}.png"
                output_path = os.path.join(output_dir, output_filename)
                cropped.save(output_path)
                names.append(out_path if return_paths else filename)

        return names

    except FileNotFoundError:
        print(f"X Error: No se encontró el archivo '{input_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"X Error al procesar la imagen: {str(e)}")
        sys.exit(1)
