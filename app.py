import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    st.set_page_config(page_title='Chat with multi PDF', page_icon=":books")

    st.header("Chat with multi PDFs :books:")
    st.text_input("Ask any question from the PDFs")

    with st.sidebar:
        st.subheader("your docs")
        pdf_docs = st.file_uploader("upload your PDFs here and click on process", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)





if __name__ == '__main__':
    main()