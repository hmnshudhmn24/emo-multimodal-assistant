# app.py - Streamlit demo for Emo Multimodal Assistant
import streamlit as st
from inference import EmoAssistant
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Emo Assistant", layout="centered")
st.title("Emo — Multimodal Emotion-Aware Assistant")

st.markdown("This demo detects emotion from text and optional image, then generates an empathetic response.")

# model selection / paths
text_model = st.text_input("Text emotion model (HF repo or local path)", value="distilbert-base-uncased")
response_model = st.text_input("Response generator model (HF repo or local path)", value="t5-small")

assistant = None
if st.button("Load models"):
    with st.spinner("Loading models — this may take a minute..."):
        assistant = EmoAssistant(text_emotion_model=text_model, response_model=response_model)
    st.success("Models loaded — ready to go!")

user_text = st.text_area("Your message", value="I had a rough day at work and feel exhausted.", height=120)

uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg","jpeg","png"])
image_path = None
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.flush()
    image_path = tfile.name
    st.image(Image.open(image_path), caption="Uploaded image", use_column_width=True)

if st.button("Get empathetic reply"):
    if assistant is None:
        with st.spinner("Loading models (first time)..."):
            assistant = EmoAssistant(text_emotion_model=text_model, response_model=response_model)
    with st.spinner("Detecting emotion and generating response..."):
        reply = assistant.respond(user_text, image_path=image_path)
    st.subheader("Assistant reply")
    st.write(reply)
