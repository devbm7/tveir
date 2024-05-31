import streamlit as st
from transformers import pipeline
from PIL import Image


st.header(':red[2]',divider='violet')

st.subheader('Hotdog or Not Hotdog?')
pipeline = pipeline(task='image-classification', model='julien-c/hotdog-not-hotdog')
file_name = st.file_uploader("Upload a hotdog candidate image")

if file_name is not None:
    col1, col2 = st.columns(2)
    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = pipeline(image)
    col2.header("Probabilities")
    for p in predictions:
        col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")