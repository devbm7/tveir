import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import io

st.header('5 - Image-To-AudioStory', divider='violet')

# Ensure the model and processor are loaded outside the function to avoid reloading them on every run
@st.cache(allow_output_mutation=True)
def load_image_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def generate_caption(processor, model, image, text=None):
    if text:
        inputs = processor(image, text, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")
    
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

processor, model = load_image_captioning_model()

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Conditional image captioning
text = "a photography of"
conditional_caption = generate_caption(processor, model, raw_image, text)
st.write(f"Conditional caption: {conditional_caption}")

# Unconditional image captioning
unconditional_caption = generate_caption(processor, model, raw_image)
st.write(f"Unconditional caption: {unconditional_caption}")

## Text-to-Speech

# Load T5-based Text-to-Speech models
processor_tts = SpeechT5Processor.from_pretrained("microsoft/speecht5-tts")
model_tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5-tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5-hifigan")

# Prepare inputs for Text-to-Speech
inputs_tts = processor_tts(text="Hello, my dog is cute.", return_tensors="pt")

# Load x-vector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Generate speech
speech = model_tts.generate_speech(inputs_tts["input_ids"], speaker_embeddings, vocoder=vocoder)

# Save speech to a file
file_obj = io.BytesIO()
sf.write(file_obj, speech.numpy(), samplerate=22050, format='wav')

# Display the audio file
st.audio(file_obj, format='audio/wav')
