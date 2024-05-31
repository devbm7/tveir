import streamlit as st
import importlib.util
import os

st.set_page_config(
    page_title="Tveir",
    page_icon=":fog:",
)

st.sidebar.markdown('Main Page :balloon:')
st.title(':blue[Tveir]')
st.write('This is the index page.')
st.markdown('this is essentially a one-size-fits-all project.')

# Dictionary to map page names to file paths
pages = {
    "Home": None,
    # "Page A": "pages/1_a.py",
    "Hotdog Page": "pages/2_hotdog.py",
    "HuggingFace Tutorial": "pages/3_hf_tutorial.py",
    "All place": "pages/4_all.py",
    "Uber from Documentation": "pages/5_Uber_from_doc.py"
}


# Selectbox for navigation
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Function to load and execute a page script
def load_page(page_path):
    if page_path is not None and os.path.exists(page_path):
        spec = importlib.util.spec_from_file_location("page_module", page_path)
        page_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(page_module)

# Display the selected page
if selected_page in pages:
    load_page(pages[selected_page])
