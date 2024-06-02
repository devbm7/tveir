import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import io

st.header('5 - Image-To-AudioStory', divider='violet')

# Load models outside the function for caching
@st.cache(allow_output_mutation=True)
def load_models():
    processor_image = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model_image = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    processor_tts = SpeechT5Processor.from_pretrained("microsoft/speecht5-tts")
    model_tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5-tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5-hifigan")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return processor_image, model_image, processor_tts, model_tts, vocoder, speaker_embeddings

def generate_caption(processor, model, image, text=None):
    if text:
        inputs = processor(image, text, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")
    
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_speech(processor_tts, model_tts, vocoder, text, speaker_embeddings):
    inputs_tts = processor_tts(text=text, return_tensors="pt")
    speech = model_tts.generate_speech(inputs_tts["input_ids"], speaker_embeddings, vocoder=vocoder)
    return speech

# Load all models
processor_image, model_image, processor_tts, model_tts, vocoder, speaker_embeddings = load_models()

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Conditional image captioning
text = "a photography of"
conditional_caption = generate_caption(processor_image, model_image, raw_image, text)
st.write(f"Conditional caption: {conditional_caption}")

# Unconditional image captioning
unconditional_caption = generate_caption(processor_image, model_image, raw_image)
st.write(f"Unconditional caption: {unconditional_caption}")

# Text-to-Speech
speech = generate_speech(processor_tts, model_tts, vocoder, "Hello, my dog is cute.", speaker_embeddings)

# Use st.experimental_memo to cache audio data separately
@st.experimental_memo
def get_audio_bytes(speech):
    file_obj = io.BytesIO()
    sf.write(file_obj, speech.numpy(), samplerate=22050, format='wav')
    return file_obj.getvalue()

audio_bytes = get_audio_bytes(speech)
st.audio(audio_bytes, format='audio/wav')