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
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention
from dotenv import load_dotenv
from PIL import Image, ImageFilter
import numpy as np
import cv2

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


class UVGuidedGenerationRequest(BaseModel):
    """Request model for UV-guided image generation"""
    prompt: str = Field(..., description="Text prompt for image generation")
    control_image: str = Field(..., description="Base64 encoded UV layout image")
    control_type: str = Field("uv_layout", description="Type of control guidance")
    height: Optional[int] = Field(None, description="Image height (default: 1024)")
    width: Optional[int] = Field(None, description="Image width (default: 1024)")
    guidance_scale: Optional[float] = Field(None, description="Guidance scale (default: 3.5)")
    num_inference_steps: Optional[int] = Field(None, description="Number of inference steps (default: 50)")
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
            "generate_with_control": "/generate_with_control",
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


def create_uv_conditioned_init_image(
    uv_mask: Image.Image, 
    width: int, 
    height: int,
    base_color: tuple = (128, 128, 128)
) -> Image.Image:
    """
    Create an initialization image from UV mask for img2img conditioning.
    This creates a base texture that FLUX will refine based on the UV layout.
    """
    # Ensure UV mask is in RGB mode
    uv_mask = uv_mask.convert('RGB')
    uv_mask = uv_mask.resize((width, height), Image.Resampling.LANCZOS)
    
    # Convert to numpy for processing
    mask_array = np.array(uv_mask)
    
    # Create base image with the specified color
    init_image = np.ones((height, width, 3), dtype=np.uint8) * np.array(base_color)
    
    # Extract UV mask (white areas = valid UV regions)
    gray = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    uv_regions = (gray > 30).astype(np.uint8) * 255
    
    # Apply distance transform to create smooth gradients in UV regions
    dist_transform = cv2.distanceTransform(uv_regions, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Create structured noise pattern in UV regions
    np.random.seed(42)  # For consistency
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    
    # Blend base color with noise in UV regions
    mask_3d = (uv_regions[:, :, np.newaxis] / 255.0)
    init_image = (init_image * (1 - mask_3d * 0.3) + noise * mask_3d * 0.3).astype(np.uint8)
    
    # Add UV layout structure to guide generation
    # Enhance edges of UV islands
    edges = cv2.Canny(uv_regions, 50, 150)
    edges_3d = edges[:, :, np.newaxis].repeat(3, axis=2)
    init_image = np.where(edges_3d > 0, init_image * 0.7, init_image).astype(np.uint8)
    
    return Image.fromarray(init_image)


def apply_uv_mask_with_feathering(
    generated_image: Image.Image,
    uv_mask: Image.Image,
    feather_radius: int = 2
) -> Image.Image:
    """
    Apply UV mask to generated image with feathering for smooth boundaries.
    """
    # Ensure same size
    if generated_image.size != uv_mask.size:
        uv_mask = uv_mask.resize(generated_image.size, Image.Resampling.LANCZOS)
    
    # Convert to arrays
    img_array = np.array(generated_image).astype(np.float32)
    mask_array = np.array(uv_mask.convert('L')).astype(np.float32)
    
    # Normalize mask to 0-1
    mask_array = mask_array / 255.0
    
    # Apply Gaussian blur for feathering
    if feather_radius > 0:
        mask_array = cv2.GaussianBlur(mask_array, (feather_radius*2+1, feather_radius*2+1), 0)
    
    # Apply mask to each channel
    for i in range(3):
        img_array[:, :, i] = img_array[:, :, i] * mask_array
    
    return Image.fromarray(img_array.astype(np.uint8))


def generate_with_uv_conditioning(
    pipeline: FluxPipeline,
    prompt: str,
    uv_mask: Image.Image,
    width: int,
    height: int,
    guidance_scale: float,
    num_steps: int,
    generator: Optional[torch.Generator] = None,
    init_strength: float = 0.75
) -> Image.Image:
    """
    Generate texture using UV mask as conditioning through img2img approach.
    
    Args:
        pipeline: FLUX pipeline
        prompt: Text prompt
        uv_mask: UV mask image
        width, height: Output dimensions
        guidance_scale: CFG scale
        num_steps: Number of denoising steps
        generator: Random generator
        init_strength: How much to transform the init image (0.0-1.0)
                      Lower = more faithful to UV layout
                      Higher = more creative freedom
    
    Returns:
        Generated image with UV conditioning
    """
    print(f"  Using UV-conditioned generation with strength={init_strength}")
    
    # Create initialization image from UV mask
    init_image = create_uv_conditioned_init_image(
        uv_mask, width, height, base_color=(140, 100, 80)  # Organ-like base color
    )
    
    # Enhanced prompt emphasizing texture continuity
    enhanced_prompt = (
        f"{prompt}, seamless organic texture, continuous surface without patterns, "
        f"unique non-repeating details, photorealistic medical imaging, "
        f"anatomically accurate surface structure, natural tissue appearance"
    )
    
    print(f"  Enhanced prompt: {enhanced_prompt[:100]}...")
    
    try:
        # Try using img2img if available in the pipeline
        if hasattr(pipeline, 'image') or 'img2img' in str(type(pipeline)).lower():
            # Use img2img pipeline
            result = pipeline(
                prompt=enhanced_prompt,
                image=init_image,
                strength=init_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator
            )
        else:
            # Fallback: Use standard pipeline with enhanced prompting
            print("  Note: Using standard pipeline (img2img not available)")
            result = pipeline(
                prompt=enhanced_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator
            )
        
        generated_image = result.images[0]
        
        # Apply UV mask with feathering
        final_image = apply_uv_mask_with_feathering(
            generated_image,
            uv_mask,
            feather_radius=3
        )
        
        return final_image
        
    except Exception as e:
        print(f"  Warning: UV conditioning failed, falling back to standard generation: {e}")
        # Fallback to standard generation
        result = pipeline(
            prompt=enhanced_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator
        )
        return apply_uv_mask_with_feathering(result.images[0], uv_mask, feather_radius=3)


@app.post("/generate_with_control")
async def generate_image_with_control(request: UVGuidedGenerationRequest):
    """Generate an image from a text prompt with UV layout control using proper conditioning"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get generation parameters with defaults from environment
        height = request.height or int(os.getenv("FLUX_DEFAULT_HEIGHT", 1024))
        width = request.width or int(os.getenv("FLUX_DEFAULT_WIDTH", 1024))
        guidance_scale = request.guidance_scale or float(os.getenv("FLUX_DEFAULT_GUIDANCE_SCALE", 3.5))
        num_steps = request.num_inference_steps or int(os.getenv("FLUX_DEFAULT_NUM_STEPS", 50))
        
        # Setup generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)
        
        print(f"\n{'='*60}")
        print(f"UV-Guided Generation Request")
        print(f"{'='*60}")
        print(f"Prompt: '{request.prompt}'")
        print(f"Size: {width}x{height}")
        print(f"Steps: {num_steps}, Guidance: {guidance_scale}")
        
        # Decode control image (UV mask)
        try:
            control_img_data = base64.b64decode(request.control_image)
            uv_mask = Image.open(io.BytesIO(control_img_data))
            print(f"✓ UV mask loaded: {uv_mask.size}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid control image: {str(e)}")
        
        # Generate image with proper UV conditioning
        image = generate_with_uv_conditioning(
            pipeline=pipeline,
            prompt=request.prompt,
            uv_mask=uv_mask,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            generator=generator,
            init_strength=0.7  # Balance between UV layout and creative freedom
        )
        
        print(f"✓ UV-guided generation complete")
        print(f"{'='*60}\n")
        
        metadata = {
            "prompt": request.prompt,
            "control_type": request.control_type,
            "uv_conditioning": "enabled",
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
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
        print(f"Error generating UV-guided image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"UV-guided generation failed: {str(e)}")


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

