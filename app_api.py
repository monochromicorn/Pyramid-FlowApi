# app_api.py

import os
import uuid
import torch
import PIL
from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
import threading
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from celery import states
from celery_app import celery_app
from celery.result import AsyncResult
import base64
from io import BytesIO
import requests

# Disabling parallelism to avoid deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize FastAPI
app = FastAPI()

# Global model cache
model_cache = {}

# Lock to ensure thread-safe access to the model cache
model_cache_lock = threading.Lock()

# Configuration
model_name = "pyramid_flux"    # or "pyramid_mmdit"
model_repo = (
    "rain1011/pyramid-flow-sd3" if model_name == "pyramid_mmdit" else "rain1011/pyramid-flow-miniflux"
)

model_dtype = "bf16"  # Supports "bf16" and "fp32"
variants = {
    "high": "diffusion_transformer_768p",  # For high-resolution version
    "low": "diffusion_transformer_384p",   # For low-resolution version
}
required_file = "config.json"  # Ensure config.json is present
width_high = 1280
height_high = 768
width_low = 640
height_low = 384
cpu_offloading = torch.cuda.is_available()  # Enable CPU offloading if CUDA is available

# Get the current working directory and create a folder to store the model
current_directory = os.getcwd()
model_path = os.path.join(current_directory, "pyramid_flow_model")  # Directory to store the model

# Download the model if not already present
def download_model_from_hf(model_repo, model_dir, variants, required_file):
    need_download = False
    if not os.path.exists(model_dir):
        print(f"[INFO] Model directory '{model_dir}' does not exist. Initiating download...")
        need_download = True
    else:
        # Check if all required files exist for each variant
        for variant_key, variant_dir in variants.items():
            variant_path = os.path.join(model_dir, variant_dir)
            file_path = os.path.join(variant_path, required_file)
            if not os.path.exists(file_path):
                print(f"[WARNING] Required file '{required_file}' missing in '{variant_path}'.")
                need_download = True
                break

    if need_download:
        print(f"[INFO] Downloading model from '{model_repo}' to '{model_dir}'...")
        try:
            snapshot_download(
                repo_id=model_repo,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                repo_type="model",
            )
            print("[INFO] Model download complete.")
        except Exception as e:
            print(f"[ERROR] Failed to download the model: {e}")
            raise
    else:
        print(f"[INFO] All required model files are present in '{model_dir}'. Skipping download.")

# Download model from Hugging Face if not present
download_model_from_hf(model_repo, model_path, variants, required_file)

# Function to initialize the model based on user options
def initialize_model(variant):
    print(f"[INFO] Initializing model with variant='{variant}', using {model_dtype} precision...")

    # Determine the correct variant directory
    variant_dir = variants["high"] if variant == "768p" else variants["low"]
    base_path = model_path  # Pass the base model path

    print(f"[DEBUG] Model base path: {base_path}")

    # Verify that config.json exists in the variant directory
    config_path = os.path.join(model_path, variant_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] config.json not found in '{os.path.join(model_path, variant_dir)}'.")
        raise FileNotFoundError(f"config.json not found in '{os.path.join(model_path, variant_dir)}'.")

    if model_dtype == "bf16":
        torch_dtype_selected = torch.bfloat16
    else:
        torch_dtype_selected = torch.float32

    # Initialize the model
    try:
        model = PyramidDiTForVideoGeneration(
            base_path,                # Pass the base model path
            model_name=model_name,    # Set to "pyramid_flux" or "pyramid_mmdit"
            model_dtype=model_dtype,  # Use "bf16" or "fp32"
            model_variant=variant_dir,  # Pass the variant directory name
            cpu_offloading=cpu_offloading,  # Pass the CPU offloading flag
        )

        # Always enable tiling for the VAE
        model.vae.enable_tiling()

        # Device placement
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            if not cpu_offloading:
                model.vae.to("cuda")
                model.dit.to("cuda")
                model.text_encoder.to("cuda")
        elif torch.backends.mps.is_available():
            model.vae.to("mps")
            model.dit.to("mps")
            model.text_encoder.to("mps")
        else:
            print("[WARNING] CUDA is not available. Proceeding without GPU.")

        print("[INFO] Model initialized successfully.")
        return model, torch_dtype_selected
    except Exception as e:
        print(f"[ERROR] Error initializing model: {e}")
        raise

# Function to get the model from cache or initialize it
def initialize_model_cached(variant, seed):
    key = variant

    if seed == 0:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Check if the model is already in the cache
    if key not in model_cache:
        with model_cache_lock:
            # Double-checked locking to prevent race conditions
            if key not in model_cache:
                model, dtype = initialize_model(variant)
                model_cache[key] = (model, dtype)

    return model_cache[key]

def resize_crop_image(img: PIL.Image.Image, tgt_width, tgt_height):
    ori_width, ori_height = img.width, img.height
    scale = max(tgt_width / ori_width, tgt_height / ori_height)
    resized_width = round(ori_width * scale)
    resized_height = round(ori_height * scale)
    img = img.resize((resized_width, resized_height), resample=PIL.Image.LANCZOS)

    left = (resized_width - tgt_width) / 2
    top = (resized_height - tgt_height) / 2
    right = (resized_width + tgt_width) / 2
    bottom = (resized_height + tgt_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))

    return img

# Pydantic models for request and response validation
class TextToVideoRequest(BaseModel):
    prompt: str
    temp: int = 16
    guidance_scale: float = 9.0
    video_guidance_scale: float = 5.0
    resolution: str = "384p"  # Options: "384p" or "768p"
    seed: int = 0  # Set to 0 for random seed
    webhook_url: Optional[str] = None  # Optional webhook URL

class ImageToVideoRequest(BaseModel):
    input_image: str  # Base64-encoded image
    prompt: str
    temp: int = 16
    video_guidance_scale: float = 4.0
    resolution: str = "384p"  # Options: "384p" or "768p"
    seed: int = 0  # Set to 0 for random seed
    webhook_url: Optional[str] = None  # Optional webhook URL

# Function to generate text-to-video
def process_text_to_video(request: TextToVideoRequest):
    variant = '768p' if request.resolution == "768p" else '384p'
    height = height_high if request.resolution == "768p" else height_low
    width = width_high if request.resolution == "768p" else width_low

    # Initialize model
    model, torch_dtype_selected = initialize_model_cached(variant, request.seed)

    # Generate video frames
    with torch.no_grad(), torch.autocast(model.device.type, dtype=torch_dtype_selected):
        frames = model.generate(
            prompt=request.prompt,
            num_inference_steps=[20, 20, 20],
            video_num_inference_steps=[10, 10, 10],
            height=height,
            width=width,
            temp=request.temp,
            guidance_scale=request.guidance_scale,
            video_guidance_scale=request.video_guidance_scale,
            output_type="pil",
            cpu_offloading=cpu_offloading,
            save_memory=True,
        )

    # Save video to a temporary path
    video_path = f"{str(uuid.uuid4())}_text_to_video_sample.mp4"
    export_to_video(frames, video_path, fps=24)

    # Read the video file and encode it in base64
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        encoded_video = base64.b64encode(video_bytes).decode("utf-8")

    # Clean up the temporary video file
    os.remove(video_path)

    return {"video": encoded_video, "seed": request.seed}

# Function to generate image-to-video
def process_image_to_video(request: ImageToVideoRequest):
    variant = '768p' if request.resolution == "768p" else '384p'
    height = height_high if request.resolution == "768p" else height_low
    width = width_high if request.resolution == "768p" else width_low

    # Decode input image
    input_image_data = base64.b64decode(request.input_image)
    input_image = Image.open(BytesIO(input_image_data)).convert("RGB")
    input_image = resize_crop_image(input_image, width, height)

    # Initialize model
    model, torch_dtype_selected = initialize_model_cached(variant, request.seed)

    # Generate video frames
    with torch.no_grad(), torch.autocast(model.device.type, dtype=torch_dtype_selected):
        frames = model.generate_i2v(
            prompt=request.prompt,
            input_image=input_image,
            num_inference_steps=[10, 10, 10],
            temp=request.temp,
            video_guidance_scale=request.video_guidance_scale,
            output_type="pil",
            cpu_offloading=cpu_offloading,
            save_memory=True,
        )

    # Save video to a temporary path
    video_path = f"{str(uuid.uuid4())}_image_to_video_sample.mp4"
    export_to_video(frames, video_path, fps=24)

    # Read the video file and encode it in base64
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        encoded_video = base64.b64encode(video_bytes).decode("utf-8")

    # Clean up the temporary video file
    os.remove(video_path)

    return {"video": encoded_video, "seed": request.seed}

# Celery task for text-to-video generation
@celery_app.task(bind=True)
def generate_text_to_video_task(self, request_data):
    try:
        request = TextToVideoRequest(**request_data)
        result = process_text_to_video(request)
        # Send webhook if provided
        if request.webhook_url:
            requests.post(request.webhook_url, json=result)
        return result
    except Exception as e:
        print(f"[ERROR] Exception in generate_text_to_video_task: {e}")
        self.update_state(state='FAILURE', meta=str(e))
        raise

# Celery task for image-to-video generation
@celery_app.task(bind=True)
def generate_image_to_video_task(self, request_data):
    try:
        request = ImageToVideoRequest(**request_data)
        result = process_image_to_video(request)
        # Send webhook if provided
        if request.webhook_url:
            requests.post(request.webhook_url, json=result)
        return result
    except Exception as e:
        print(f"[ERROR] Exception in generate_image_to_video_task: {e}")
        self.update_state(state='FAILURE', meta=str(e))
        raise

# API endpoint to start text-to-video generation
@app.post("/text_to_video")
def text_to_video_endpoint(request: TextToVideoRequest):
    try:
        task = generate_text_to_video_task.delay(request.dict())
        return {"job_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to start image-to-video generation
@app.post("/image_to_video")
def image_to_video_endpoint(request: ImageToVideoRequest):
    try:
        task = generate_image_to_video_task.delay(request.dict())
        return {"job_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to check task status
@app.get("/status/{job_id}")
def get_status(job_id: str):
    task = AsyncResult(job_id, app=celery_app)
    if task.state == states.PENDING:
        return {"status": "PENDING"}
    elif task.state == states.STARTED:
        return {"status": "STARTED"}
    elif task.state == states.SUCCESS:
        return {"status": "SUCCESS"}
    elif task.state == states.FAILURE:
        error_message = str(task.info)
        return {"status": "FAILURE", "error": error_message}
    else:
        return {"status": str(task.state)}

# API endpoint to retrieve results
@app.get("/result/{job_id}")
def get_result(job_id: str):
    task = AsyncResult(job_id, app=celery_app)
    if task.state == states.SUCCESS:
        return task.result
    elif task.state == states.FAILURE:
        raise HTTPException(status_code=500, detail=str(task.info))
    else:
        raise HTTPException(status_code=202, detail="Task not completed yet.")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="0.0.0.0", port=8001)
