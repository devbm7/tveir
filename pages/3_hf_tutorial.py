import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import html
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
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
    # classifier = pipeline('sentiment-analysis')
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline(
        task='sentiment-analysis',
        model=model,
        tokenizer=tokenizer
    )

    x = st.text_input(label='Enter text', value="I've been waiting for a huggingface course my whoole life.")
    res = classifier(x)
    # st.markdown(body=f"*Prediction*: :green-background[{res[0]['label']}]")
    # st.markdown(f"*Score*: :green-background[{res[0]['score']}]")
    col1, col2 = st.columns(2)
    col1.metric(label='Prediction', value=res[0]['label'])
    col2.metric(label='Score', value=res[0]['score'])
    st.write(res)
    if st.checkbox(label='Show Tokenizer Explanation'):
        res = tokenizer(x)
        st.write(res)
        tokens = tokenizer.tokenize(x)
        st.write(tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        st.write((ids))
        decoded_string = tokenizer.decode(ids)
        st.write(decoded_string)
    if st.checkbox(label='show for multiple sentences and saving-loading of models'):
        X_train = [
            "I've been waiting for a huggingface course my whole life.",
            "I am happy.",
            "I lost a match today."
        ]
        res = classifier(X_train)
        st.write(res)
        batch = tokenizer(X_train,padding=True,truncation=True, max_length=512,return_tensors='pt')
        st.write(batch)
        with torch.no_grad():
            outputs = model(**batch)
            st.write(outputs)
            predictions = F.softmax(outputs.logits, dim=1)
            st.write(predictions)
            labels = torch.argmax(predictions, dim=1)
            st.write(labels)
        save_directory = 'p/'
        tokenizer.save_pretrained(save_directory=save_directory)
        model.save_pretrained(save_directory=save_directory)

        tok = AutoTokenizer.from_pretrained(save_directory)
        mod = AutoModelForSequenceClassification.from_pretrained(save_directory)

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
    slowly_display_text(res2[0]['generated_text'])
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

#######################################################
st.subheader("Pipe4 :- Image Classification", divider='orange')

if st.toggle(label='Show Pipe4'):
    models = [
        'google/vit-base-patch16-224',
        'WinKawaks/vit-tiny-patch16-224',
        'microsoft/resnet-50',
        'facebook/deit-base-distilled-patch16-224',
        'facebook/convnext-large-224',
        'apple/mobilevit-small',
    ]
    model_name = st.selectbox(
        label='Select Model',
        options=models,
        placeholder='google/vit-base-patch16-224',
    )
    pipe = pipeline("image-classification", model=model_name)
    url = 'https://media.istockphoto.com/id/182756302/photo/hot-dog-with-grilled-peppers.jpg?s=1024x1024&w=is&k=20&c=NCHo2P94a-PfRDKzWSe4h6oACQZ-_ubZqUBj5CMSEWY='
    response = requests.get(url=url)
    image_bytes = BytesIO(response.content)
    image = Image.open(image_bytes)
    # image = Image.open(BytesIO(requests.get(url).content))
    # use_default = st.checkbox(label='Use default image')
    file = st.file_uploader(label='Upload image')
    if file is not None:
        image = Image.open(file)

    res = pipe(image)
    if st.toggle(label='Show row data'):
        st.write(res)
    p = pd.DataFrame(res)
    p = p.sort_values(by='score',ascending=False)
    col1, col2 = st.columns(2)
    col1.write(image)
    col2.write(p['label'])
    st.bar_chart(p.set_index('label'))
    st.area_chart(p.set_index('label'))
    # col2.bar_chart(p.set_index('label'))


############################################################
st.subheader('Pipe5: Text-To-Text Generation -> Que. Generation',divider='orange')

if st.toggle(label='Show Pipe5'):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    input_text = st.text_area(label='Enter the text from which question is to be generated:',value='Bruce Wayne is the Batman.')
    input_text = 'Generate a question from this: ' + input_text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids


    outputs = model.generate(input_ids)
    output_text = tokenizer.decode(outputs[0][1:len(outputs[0])-1])
    if st.checkbox(label='Show Tokenized output'):
        st.write(outputs)
    st.write("Output is:")
    st.write(f"{output_text}")
    if st.toggle(label='Access model unrestricted'):
        input_text = st.text_area('Enter text')
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids
        outputs = model.generate(input_ids)
        st.write(tokenizer.decode(outputs[0]))
        st.write(outputs)