"""
Z-Image-Turbo image generation service using jb-service SDK.

Uses the Tongyi-MAI/Z-Image-Turbo model for fast, high-quality image generation.
"""
import os
import torch
from typing import Optional
from jb_service import MessagePackService, method, run, save_image


class ZImageTurbo(MessagePackService):
    """Fast image generation using Z-Image-Turbo (8-step diffusion)."""
    
    name = "z-image-turbo"
    version = "1.0.0"
    
    def setup(self):
        """Load the Z-Image-Turbo pipeline."""
        from diffusers import ZImagePipeline
        
        self.log.info("Loading Z-Image-Turbo pipeline...")
        
        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
            self.log.info("Using CUDA with bfloat16")
        else:
            device = "cpu"
            dtype = torch.float32
            self.log.warning("CUDA not available, using CPU (will be slow)")
        
        # Progress bars are fine - MessagePack transport doesn't care about stdout
        self.log.info("Loading model from HuggingFace...")
        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
        )
        
        self.log.info(f"Moving model to {device}...")
        self.pipe.to(device)
        self.device = device
        
        self.log.info(f"Z-Image-Turbo loaded on {device}")
    
    @method
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        seed: Optional[int] = None,
        format: str = "png",
        ttl: int = 3600,
    ) -> dict:
        """Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            width: Image width in pixels (default 1024)
            height: Image height in pixels (default 1024)
            steps: Number of inference steps (default 9, which gives 8 DiT forwards)
            seed: Random seed for reproducibility (optional)
            format: Output format - png, jpg, or webp (default png)
            ttl: Time to live in seconds (default 3600 = 1 hour, 0 = permanent)
        
        Returns:
            Dictionary with 'file_id' for the stored image
        """
        self.log.info(f"Generating image: {prompt[:50]}...")
        
        # Set up generator with seed
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(seed))
        
        # Generate - tqdm progress bars are fine with MessagePack transport
        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,  # Must be 0 for Turbo model
            generator=generator,
        )
        
        image = result.images[0]
        
        # Save to temp file
        temp_path = save_image(image, format=format)
        
        # Import into file store with TTL
        seed_str = str(int(seed)) if seed is not None else "random"
        file_name = f"z-image-{seed_str}.{format}"
        file_id = self.files.import_file(temp_path, name=file_name, ttl=ttl)
        
        # Clean up temp file
        os.remove(temp_path)
        
        self.log.info(f"Generated image stored as {file_id}")
        
        return {
            "file_id": file_id,
            "width": width,
            "height": height,
            "prompt": prompt,
            "seed": seed,
            "ttl": ttl,
        }
    
    @method
    def health(self) -> dict:
        """Health check."""
        return {
            "status": "ok",
            "device": self.device,
            "model": "Z-Image-Turbo",
        }


if __name__ == "__main__":
    run(ZImageTurbo)
