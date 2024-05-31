import streamlit as st
import numpy as np
import pandas as pd
import time

st.header(body=':red[4]', divider='violet')
st.sidebar.markdown('Page 4 :snowflake:')
if st.checkbox(label='Show DataFrame'):
    chart_data = pd.DataFrame(
        np.random.randn(20,3),
        columns=['a','b','c']
    )
    chart_data

if st.checkbox(label='Show time progess bar'):
    'Starting a long computation'

    # add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration
        latest_iteration.text(f"Iteration {i+1}")
        bar.progress(i + 1)
        time.sleep(0.2)

    '.. and now we\'re done!'

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.write(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")