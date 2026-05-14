import os
import torch
import math
import matplotlib.pyplot as plt
from diffusers import StableDiffusion3Pipeline

# 1. Load the SD 3.5 Medium Pipeline
model_id = "stabilityai/stable-diffusion-3.5-medium"

# Note: We use bfloat16 as it is much safer against overflow in SD 3.5
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    use_safetensors=True,
    token="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" # Uncomment and add your token if not logged in via terminal
)

# SD 3.5 is heavy! This offloads inactive components to standard RAM to save VRAM.
# This replaces `pipe.to("cuda")`
pipe.enable_model_cpu_offload()

# List to store our intermediate images
intermediate_images = []

# 2. Define the callback function
def decode_step_callback(pipe, step_index, timestep, callback_kwargs):
    # Retrieve the latents from the current step
    latents = callback_kwargs["latents"]
    
    with torch.no_grad():
        # SD 3.5 VAE requires BOTH scaling and shifting
        scaled_latents = latents / pipe.vae.config.scaling_factor
        if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor is not None:
            scaled_latents = scaled_latents + pipe.vae.config.shift_factor
        
        # Decode into pixel space
        image_tensor = pipe.vae.decode(scaled_latents, return_dict=False)[0]
        
        # Post-process the tensor to a PIL image
        image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
        
        # Save the image and the step number
        intermediate_images.append((step_index + 1, image))
        
    return callback_kwargs

# 3. Generate the Image
prompt = "A highly detailed oil painting of a futuristic city skyline at sunset, cyberpunk style"

print("Generating image and decoding intermediate steps...")
final_image = pipe(
    prompt=prompt,
    num_inference_steps=15, 
    guidance_scale=4.5, # SD 3.5 prefers lower guidance scales than SDXL
    callback_on_step_end=decode_step_callback,
    callback_on_step_end_tensor_inputs=["latents"] 
).images[0]

# Add the final fully processed image to the end of our list for comparison
intermediate_images.append(("Final", final_image))

# 4. Plot the images in a grid using Matplotlib
num_images = len(intermediate_images)
cols = 4
rows = math.ceil(num_images / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten()

for i, (step_label, img) in enumerate(intermediate_images):
    axes[i].imshow(img)
    axes[i].set_title(f"Step: {step_label}")
    axes[i].axis("off")

# Hide any empty subplots
for i in range(num_images, len(axes)):
    axes[i].axis("off")

plt.tight_layout()

# 5. Save the image to the "test" folder
output_folder = "test"
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, "sd35_diffusion_steps.png")

plt.savefig(save_path, bbox_inches="tight")
print(f"\nImage successfully saved to: {save_path}")

plt.show()