import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import nltk

nltk.download('punkt')

st.title(body='7 - Question Generation')


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


########################################################
st.subheader(body='Proposition 1',divider='orange')

if st.toggle(label='Show Proposition 1'):
    st.title('Question Generator from PDFs')
    if st.checkbox('Show Caption'):
        st.caption('Hugging Face Model used: ramsrigouthamg/t5_squad_v1')
    pipe = pipeline(
        task = 'text2text-generation',
        model = 'ramsrigouthamg/t5_squad_v1'
    )
    file = st.file_uploader(label='Upload',accept_multiple_files=True)
    pr = st.button(label='Process')
    if pr:
        with st.spinner('Processing'):
            raw_text = get_pdf_text(file)
            sentences = nltk.sent_tokenize(text=raw_text)
            s = pipe(sentences)
            questions = []
            
        st.subheader("*Generated Questions are:*")
        
        for i in s:
            x = i['generated_text'][10:]
            questions.append(x)
            st.write(f':blue[{x}]')
        if st.toggle(label='Show Pipeline Output'):
            st.write(s)
        if st.toggle(label='Show Questions list'):
            st.write(questions)
