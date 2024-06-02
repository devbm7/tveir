import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModel
from transformers import pipeline
import torch

st.header('5 - Image-To-AudioStory', divider='violet')

# Ensure the model and processor are loaded outside the function to avoid reloading them on every run
@st.cache(allow_output_mutation=True)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model()

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Define a function for generating captions
def generate_caption(image, text=None):
    if text:
        inputs = processor(image, text, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")
    
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Conditional image captioning
text = "a photography of"
conditional_caption = generate_caption(raw_image, text)
st.write(f"Conditional caption: {conditional_caption}")

# Unconditional image captioning
unconditional_caption = generate_caption(raw_image)
st.write(f"Unconditional caption: {unconditional_caption}")

## llm - story gen

## text2speech
audio_pipeline = pipeline(
    task = 'text-to-speech',
)

audio_output = audio_pipeline(unconditional_caption)[0]['audio']
st.audio(audio_output, format='audio/wav')
# audio_output = audio_pipeline(unconditional_caption)
# st.audio(data=audio_output)