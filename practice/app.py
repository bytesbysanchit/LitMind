import streamlit as st
import fitz

# st.title("LitMind")
# st.write("LitMind is an AI-powered Novel Analyzer that uses Natural Language Processing (NLP) and Generative AI to analyze literary texts and provide meaningful insights.")
# st.header("features")
# st.subheader("• Character Extraction")
# st.subheader("• Relationship Analysis")
# st.subheader("• Character Network")
# st.subheader("• Sentiment Analysis")



# uploaded_file = st.file_uploader("Upload Novel", type=["pdf", "txt"])
# if uploaded_file is not None:
#   st.write(uploaded_file.name, uploaded_file.type)



st.title("LitMind")
st.write("AI-Powered Novel Analyzer")

uploaded_file = st.file_uploader("Upload Novel", type=["pdf"])

if uploaded_file is None:
  st.info("Please upload a PDF first.")

else:
  st.write("Uploaded file:", uploaded_file.name)

  if st.button("Analyze Novel"):
    st.write("Analysis Started!")

    pdf = fitz.open(
      stream=uploaded_file.getvalue(),
      filetype="pdf"
    )

    text = ""

    for page in pdf:
      text += page.get_text()

    pdf.close()

    st.success("Text extracted successfully!")
    st.write(text[:3000])