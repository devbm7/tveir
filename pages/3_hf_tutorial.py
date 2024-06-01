import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline

st.title('3 - *HuggingFace* :blue[Tutorial]')

######################################################
st.subheader('Pipe1',divider='orange')

if st.checkbox(label='Show Pipe1'):
    classifier = pipeline('sentiment-analysis')

    sentence = "I've been waiting for a huggingface course my whole life."
    res = classifier(sentence)

    st.write(res)
    st.write("Given Sentence is :",sentence)
    st.write("Prediction for this: ", res[0]['label'])
    st.write("Score for this: ", res[0]['score'])

######################################################
st.subheader('Pipe2',divider='orange')

if st.checkbox(label='Show Pipe2'):
    generator = pipeline('text-generation', model='distilgpt2')
    sentence = "In this course we'll teach you how to"
    res2 = generator(
        sentence,
        max_length = 30,
    )
    st.write(res2)
    st.write("Prompt is: ", sentence)
    st.write("Generated text is: ")
    st.write(res2[0]['generated_text'])


######################################################
st.subheader('Pipe3', divider='orange')

if st.checkbox(label='Show Pipe3'):
    clf2 = pipeline(
        task='zero-shot-classification',
        model = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
        framework='pt'
    )
    sentence = "This is a course about python list comprehension"
    res3 = clf2(
        sentence,
        candidate_labels = ['education', 'politics', 'business']
    )
    st.write("Text is: ", sentence)
    st.write(res3)
