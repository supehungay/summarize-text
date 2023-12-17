import streamlit as st
from Module.Summarization import *
from Module.v2_summary_tf_vi_2 import *
def PreTrainedModel(model_path:str=''):
    model = PretrainedSummary()
    model.load_from_checkpoint(model_path)
    model.freeze()
    return model

def ManualTrainedModel():
    pass

def main():
    st.set_page_config(
    page_title="Text Summarization Web App",
    page_icon="✨",
    layout="wide"
    )

    MODEL_PATH = "/content/drive/MyDrive/Saved/Model/best-checkpoint-vi3.ckpt"
    model = PreTrainedModel(MODEL_PATH)

    # File upload
    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "docx"])

    # Main content
    st.write(
        "Welcome to the Text Summarization Web App! Enter your text in the textbox below or upload a text file."
    )

    # Text input area
    text_input = st.text_area("Or enter text here:")

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the uploaded file
        file_contents = uploaded_file.read().decode('utf-8')
        # Display the content of the file
        st.subheader("Uploaded File Content:")
        st.write(file_contents)
        col1, col2 = st.columns(2) 
        # Use the content for summarization
        text_input = process_text(file_contents)
        st.subheader("Text Processed: ")
        st.write(text_input)
        with col1:
          summary1 = model.summarize(text_input)
          # Display the summary
          st.header("Summary with Pretrained-Model:")
          st.write(summary1)
        with col2:
          summary2 = summarize(text_input)
          st.header("Summary with Manual Model")
          st.write(summary2)
    elif st.button("Summarize"):
        col1, col2 = st.columns(2) 
        # Summarize the text
        text_input = process_text(text_input)
        st.subheader("Text Processed: ")
        st.write(text_input)
        with col1:
          summary1 = model.summarize(text_input)
          # Display the summary
          st.header("Summary with Pretrained-Model:")
          st.write(summary1)
        with col2:
          summary2 = summarize(text_input)
          st.header("Summary with Manual Model")
          st.write(summary2)
    # Footer
    st.markdown("---")
    st.write("Developed by HVAH")

# This line is optionasl but can improve app performance
if __name__ == "__main__":
    main()
