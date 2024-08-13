import torch
from diffusers import FluxPipeline

# Load the pipeline with mixed precision
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.float16
)

# Enable CPU offloading to save VRAM
pipe.enable_model_cpu_offload()

# Generate the image
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=768,  # Consider using a smaller resolution, e.g., 768x768 instead of 1024x1024
    width=768,
    guidance_scale=3.5,
    num_inference_steps=30,  # Reduce the number of inference steps
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(0)  # Use GPU for generation
).images[0]

# Save the image
image.save("flux-dev.png")
