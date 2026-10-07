import streamlit as st

st.title("LitMind")
st.write("LitMind is an AI-powered Novel Analyzer that uses Natural Language Processing (NLP) and Generative AI to analyze literary texts and provide meaningful insights.")
st.header("features")
st.subheader("• Character Extraction")
st.subheader("• Relationship Analysis")
st.subheader("• Character Network")
st.subheader("• Sentiment Analysis")

button= st.button('Analyze Novel')
if button:
  st.write("Analysis Started!")

uploaded_file = st.file_uploader("Upload Novel", type=["pdf", "txt"])
if uploaded_file is not None:
  st.write(uploaded_file.name, uploaded_file.type)