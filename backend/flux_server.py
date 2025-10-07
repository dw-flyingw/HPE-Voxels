"""
FLUX.1-dev Text-to-Image API Server
Serves the FLUX.1-dev model via FastAPI on Nvidia H200 GPUs
"""

import os
import io
import base64
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from diffusers import FluxPipeline
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Global variable to store the pipeline
pipeline = None


class GenerationRequest(BaseModel):
    """Request model for image generation"""
    prompt: str = Field(..., description="Text prompt for image generation")
    height: Optional[int] = Field(None, description="Image height (default: 1024)")
    width: Optional[int] = Field(None, description="Image width (default: 1024)")
    guidance_scale: Optional[float] = Field(None, description="Guidance scale (default: 3.5)")
    num_inference_steps: Optional[int] = Field(None, description="Number of inference steps (default: 50)")
    max_sequence_length: Optional[int] = Field(None, description="Max sequence length (default: 512)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    return_base64: Optional[bool] = Field(False, description="Return image as base64 string instead of PNG")


class GenerationResponse(BaseModel):
    """Response model for successful generation"""
    success: bool = True
    message: str = "Image generated successfully"
    image_base64: Optional[str] = None
    metadata: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global pipeline
    
    # Startup: Load the model
    print("Loading FLUX.1-dev model...")
    
    # Check for Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token or hf_token == "your_token_here":
        raise ValueError(
            "HUGGINGFACE_TOKEN not found or not set in .env file. "
            "Please get your token from https://huggingface.co/settings/tokens "
            "and set it in the .env file"
        )
    
    model_name = os.getenv("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-dev")
    torch_dtype_str = os.getenv("FLUX_TORCH_DTYPE", "bfloat16")
    torch_dtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float16
    
    try:
        # Load the pipeline to GPU
        # H200 has 141GB VRAM - more than enough for FLUX (~24GB needed)
        print("✓ Loading model to GPU (full GPU mode)")
        
        pipeline = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            token=hf_token
        )
        
        # Move to CUDA
        pipeline = pipeline.to("cuda")
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Using dtype: {torch_dtype}")
        print(f"✓ Device: cuda")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down server...")
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()
    print("✓ Cleanup complete")


# Initialize FastAPI app
app = FastAPI(
    title="FLUX.1-dev API Server",
    description="Text-to-Image generation using FLUX.1-dev model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": os.getenv("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-dev"),
        "device": str(pipeline.device) if pipeline else "not loaded",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": str(pipeline.device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate an image from a text prompt"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get generation parameters with defaults from environment
        height = request.height or int(os.getenv("FLUX_DEFAULT_HEIGHT", 1024))
        width = request.width or int(os.getenv("FLUX_DEFAULT_WIDTH", 1024))
        guidance_scale = request.guidance_scale or float(os.getenv("FLUX_DEFAULT_GUIDANCE_SCALE", 3.5))
        num_steps = request.num_inference_steps or int(os.getenv("FLUX_DEFAULT_NUM_STEPS", 50))
        max_seq_length = request.max_sequence_length or int(os.getenv("FLUX_DEFAULT_MAX_SEQ_LENGTH", 512))
        
        # Setup generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)
        
        print(f"Generating image with prompt: '{request.prompt}'")
        
        # Generate image
        image = pipeline(
            request.prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            max_sequence_length=max_seq_length,
            generator=generator
        ).images[0]
        
        metadata = {
            "prompt": request.prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "max_sequence_length": max_seq_length,
            "seed": request.seed
        }
        
        # Return as base64 or PNG
        if request.return_base64:
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return GenerationResponse(
                image_base64=img_str,
                metadata=metadata
            )
        else:
            # Return as PNG image
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            
            return StreamingResponse(
                img_byte_arr,
                media_type="image/png",
                headers={"X-Generation-Metadata": str(metadata)}
            )
            
    except Exception as e:
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("FLUX_SERVER_PORT", 8000))
    host = os.getenv("FLUX_HOST", "0.0.0.0")
    
    print(f"Starting FLUX.1-dev server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

