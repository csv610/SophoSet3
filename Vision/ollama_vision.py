import os
import io
import time
import base64
import subprocess
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import ollama
import argparse

class ImageUtils:
    @staticmethod
    def resize_image(image, max_dimension):
        width, height = image.size
        aspect_ratio = width / height
        if width > height:
            new_width = min(width, max_dimension)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(height, max_dimension)
            new_width = int(new_height * aspect_ratio)
        return image.resize((new_width, new_height))

    @staticmethod
    def square_image(image):
        width, height = image.size
        max_dim = max(width, height)
        new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        paste_position = ((max_dim - width) // 2, (max_dim - height) // 2)
        new_image.paste(image, paste_position)
        return new_image

class ImageReader:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".avif", ".heif", ".heic"}

    @staticmethod
    def convert_to_png(image_path):
        converted_path = image_path.rsplit('.', 1)[0] + ".png"
        try:
            subprocess.run(
                ["magick", image_path, converted_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            with open(converted_path, "rb") as img_file:
                image_data = img_file.read()
            os.remove(converted_path)
            return image_data
        except FileNotFoundError:
            raise RuntimeError("ImageMagick ('magick') not found. Please install or ensure it is in PATH.")
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Image conversion failed for {image_path}")

    @staticmethod
    def get_data(image_input):
        if isinstance(image_input, str):
            if image_input.startswith("data:image/"):
                try:
                    header, encoded = image_input.split(",", 1)
                    return base64.b64decode(encoded)
                except ValueError:
                    raise ValueError("Invalid Base64-encoded image data.")
            if image_input.startswith("http"):
                response = requests.get(image_input)
                if response.status_code != 200:
                    raise ValueError(f"Failed to download image from {image_input}")
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith("image/"):
                    raise ValueError(f"URL does not point to an image: Content-Type {content_type}")
                return response.content
            ext = os.path.splitext(image_input)[1].lower()
            if ext not in ImageReader.valid_extensions:
                raise ValueError(f"Unsupported file extension '{ext}'. Supported formats: {', '.join(ImageReader.valid_extensions)}")
            if ext in {".webp", ".avif", ".heif", ".heic"}:
                return ImageReader.convert_to_png(image_input)
            with open(image_input, "rb") as img_file:
                return img_file.read()

        elif isinstance(image_input, Image.Image):
            img_byte_arr = io.BytesIO()
            image_input.save(img_byte_arr, format="PNG")
            return img_byte_arr.getvalue()

        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = (255 * image_input).clip(0, 255).astype(np.uint8)
            if image_input.ndim == 2:
                img = Image.fromarray(image_input, mode="L")
            elif image_input.shape[-1] == 3:
                img = Image.fromarray(image_input, mode="RGB")
            elif image_input.shape[-1] == 4:
                img = Image.fromarray(image_input, mode="RGBA")
            else:
                raise ValueError("Unsupported NumPy image format.")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            return img_byte_arr.getvalue()

        raise ValueError("Unsupported image input format.")

class OllamaVision:
    MODELS = [
        'bakllava', 'gemma3', 'gemma3n', 'granite3.2-vision',
        'llava', 'llama3.2-vision', 'llava-llama3',
        'llava-phi3', 'minicpm-v', 'moondream:1.8b', "qwen2.5vl:7b"
    ]

    def get_response(self, question, image_input, model_name = "llama3.2-vision"):
        if model_name not in self.MODELS:
            return {
                "model": model_name,
                "text": f"Error: Unsupported model '{model_name}'. Available models: {', '.join(self.MODELS)}",
                "execution_time": 0,
                "word_count": 0
            }

        try:
            start_time = time.time()
            images = []
            if isinstance(image_input, list):
                images = [ImageReader.get_data(img) for img in image_input]
            else:
                images.append(ImageReader.get_data(image_input))

            messages = [{
                'role': 'user',
                'content': question,
                'images': images
            }]

            res = ollama.chat(model=model_name, messages=messages)
            end_time = time.time()

            text_content = res['message']['content']
            return {
                "model": model_name,
                "text": text_content,
                "execution_time": round(end_time - start_time, 3),
                "word_count": len(text_content.split())
            }
        except Exception as e:
            return {
                "model": model_name,
                "text": f"Error: {e}",
                "execution_time": 0,
                "word_count": 0
            }

def main():
    parser = argparse.ArgumentParser(description="Ollama Vision Image Questioning CLI")
    parser.add_argument("-i", "--image", type=str, required=True, help="Image path, URL, or base64 string")
    parser.add_argument("-q", "--question", type=str, required=True, help="Question to ask the model")
    parser.add_argument("-m", "--model", type=str, default="llama3.2-vision", help="Model name")
    args = parser.parse_args()

    vision = OllamaVision()
    result = vision.get_response(args.question, args.image, args.model)

    print("\n--- Ollama Vision Result ---")
    print(f"Model: {result['model']}")
    print(f"Execution Time: {result['execution_time']} seconds")
    print(f"Word Count: {result['word_count']}")
    print("Response:\n")
    print(result["text"])
    print("\n----------------------------")

if __name__ == "__main__":
    main()

