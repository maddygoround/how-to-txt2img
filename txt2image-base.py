import gradio as gr
import torch
import logging
from torch import autocast
from diffusers import DiffusionPipeline

modelid = "dreamlike-art/dreamlike-diffusion-1.0"

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
pipe = DiffusionPipeline.from_pretrained(modelid,torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe.to(device)

if device.type == "mps":
    logging.info("Enabling attention slicing for MPS")
    pipe.enable_attention_slicing()
    prompt = "a photo of an astronaut riding a horse on mars"
    _ = pipe(prompt, num_inference_steps=2)

seed = 1330

def generate(text, text_neg):
    """
    Generates an image based on the given text and negative text.

    Args:
        text (str): The main text prompt for generating the image.
        text_neg (str): The negative text prompt for generating the image.

    Returns:
        PIL.Image.Image: The generated image.
    """
    if device.type == "mps":
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device).manual_seed(seed)
    with autocast(device_type=device.type, enabled=de):
        image = pipe(prompt=text, guidance_scale=8.5, negative_prompt=text_neg, generator=generator).images[0]
    return image

with gr.Blocks() as demo:
    image_output = gr.Image(label="Output Image",width=512, height=512)
    prompt = gr.Textbox(value="A painting of a cat, high resolution",label="Positive Prompt",placeholder="A painting of a cat, high resolution")
    prompt_neg = gr.Textbox(value="bad eyes, bad ears, bad legs",label="Negative Prompt",placeholder="bad eyes, bad ears, bad legs")
    btn = gr.Button("Generate")
    btn.click(generate, inputs=[prompt,prompt_neg], outputs=image_output)

demo.launch(share=False)