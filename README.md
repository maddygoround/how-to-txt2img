# Getting Started with Stable Diffusion: Txt2Img Example

## Introduction

This project demonstrates how to set up a text-to-image (txt2img) model based on stable diffusion using the Gradio interface. It serves as an example for getting started with stable diffusion and showcases 
its capabilities in generating images from textual descriptions.

## Setup
To run this project, you'll need to follow these steps:

1. **Install Python dependencies**: Make sure you have Python 3 installed on your system. Then create a virtual environment and install the required packages by running the following command in your 
terminal or command prompt:
```bash
pip install -r requirements.txt
```
2. **Run the code**: After setting up everything, you can run the script by executing the following command:
```bash
python app.py
```
This will start the Gradio interface, allowing you to input text and generate images using stable diffusion.

## Usage

Once you've executed the code and the Gradio interface is running, follow these steps:

1. **Input your positive prompt**: Use the Gradio interface to enter a textual description of the image you want to generate.

2. **Input your negative prompt**: Use the Gradio interface to enter a textual description for negative prompt. This parameter should be a string containing a textual description of what you want to exclude or avoid in the generated image. The model will then take this into account during the generation process.

3. **Generate image**: After entering your desired input, click on the "Generate" button to start generating images based on your text description. The process may take some time depending on the 
complexity of the image and the selected model.
