import torch
from torch import autocast
import logging
from diffusers import DiffusionPipeline

modelid = "stabilityai/stable-diffusion-xl-base-1.0"
refinerid = "stabilityai/stable-diffusion-xl-refiner-1.0"

def get_device():
    """
    Get the device to be used for computation.

    Returns:
        str: The device name.
    """
    switch_cases = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }

    for case, condition in switch_cases.items():
        if condition:
            logging.info(f"Using {case} device")
            return case

    return "cpu"

device = torch.device(get_device())
pipeline = DiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, use_safetensors=True, variant="fp16")    
pipeline.to(device.type)
refiner = DiffusionPipeline.from_pretrained(refinerid, text_encoder_2=pipeline.text_encoder_2, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, use_safetensors=True, variant="fp16" , vae = pipeline.vae)
refiner.enable_sequential_cpu_offload()
refiner.enable_attention_slicing("max")
# pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)


if device.type == "mps":
    logging.info("Enabling attention slicing for MPS")
    pipeline.enable_attention_slicing()
    pipeline.enable_model_cpu_offload()
    prompt = "a photo of an astronaut riding a horse on mars"
    _ = pipeline(prompt, num_inference_steps=2)

seed = 1330
if device.type == "mps":
    generator = torch.manual_seed(seed)
else:
    generator = torch.Generator(device.type).manual_seed(seed)
image = pipeline(prompt="a photo of an astronaut riding a horse on mars",guidance_scale=8.5, generator=generator).images[0]
image = refiner(prompt="a photo of an astronaut riding a horse on mars",guidance_scale=8.5, image=image, generator=generator).images[0]
image.save("output.png")