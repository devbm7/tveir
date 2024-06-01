import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import time
import html

st.title('3 - *HuggingFace* :blue[Tutorial]')

def slowly_display_text(text, delay=0.05):
    # Define the CSS for the text container
    css = """
    <style>
    .text-container {
        width: 80%;
        max-width: 600px;
        white-space: pre-wrap; /* Ensure text wraps */
        word-wrap: break-word; /* Ensure long words wrap */
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.1em;
        line-height: 1.5;
    }
    </style>
    """
    
    # Create a placeholder for the text
    placeholder = st.empty()
    displayed_text = ""
    
    # Iterate over each character and update the text incrementally
    for char in text:
        displayed_text += html.escape(char)  # Escape HTML special characters
        # Replace newlines with <br> tags to handle empty lines correctly
        formatted_text = displayed_text.replace("\n", "<br>")
        placeholder.markdown(css + f'<div class="text-container">{formatted_text}</div>', unsafe_allow_html=True)
        time.sleep(delay)

######################################################
st.subheader('Pipe1 :- Sentiment Analysis',divider='orange')

if st.checkbox(label='Show Pipe1'):
    classifier = pipeline('sentiment-analysis')

    x = st.text_input(label='Enter text', value="I've been waiting for a huggingface course my whoole life.")
    res = classifier(x)
    # st.markdown(body=f"*Prediction*: :green-background[{res[0]['label']}]")
    # st.markdown(f"*Score*: :green-background[{res[0]['score']}]")
    col1, col2 = st.column(2)
    col1.metric(label='Prediction', value=res[0]['label'])
    col2.metric(label='Score', value=res[0]['score'])
    st.write(res)

######################################################
st.subheader('Pipe2 :- Text Generation',divider='orange')

if st.checkbox(label='Show Pipe2'):
    generator = pipeline('text-generation', model='distilgpt2')
    sentence = "In this course we'll teach you how to"
    res2 = generator(
        sentence,
        max_length = 30,
    )
    x = st.text_input(label='Enter text', value="In this course we'll teach you how to")
    res2 = generator(x,max_length=70)
    st.write("Generated text is:")
    st.write(slowly_display_text(res2[0]['generated_text']))
    st.write(res2)


######################################################
st.subheader('Pipe3 :- Zero-shot classification', divider='orange')

if st.checkbox(label='Show Pipe3'):
    clf2 = pipeline(
        task='zero-shot-classification',
        model = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
        framework='pt'
    )
    x = st.text_input(label='Enter text', value="This is a course about python list comprehension")
    res3 = clf2(
        x,
        candidate_labels = ['education', 'politics', 'business']
    )
    st.write(res3)