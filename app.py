import torch
from diffusers import FluxPipeline

# Load the pipeline and move it to the GPU
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.to("cuda")  # Move the pipeline to the GPU

# Optionally, if you have enough GPU memory, you can remove the CPU offload line
pipe.enable_model_cpu_offload()  # Comment out this line to keep the model fully on the GPU
pipe.enable_gradient_checkpointing()

prompt = "A cat holding a sign that says hello world"

# Use GPU-based generator
generator = torch.Generator("cuda").manual_seed(0)

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=generator
).images[0]

image.save("flux-dev.png")
