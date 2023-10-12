import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


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

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with multi PDF', page_icon=":books")
    st.header("Chat with multi PDFs :books:") 
        
    # Upload files
    pdf_docs = st.file_uploader("upload your PDFs here and click on process", accept_multiple_files=True)
    
    if pdf_docs is not None:
        # get pdf text
        raw_text = get_pdf_text(pdf_docs)

        # get text chunks
        text_chunks = get_text_chunks(raw_text)

        # create Vectorstore
        vectorstore = get_vectorstore(text_chunks)

        # create conversation chain
        # st.session_state.conversation=get_conversation_chain(vectorstore)

        # User question
        user_question = st.text_input("Ask a question about your PDF: ")
        if user_question:
            docs = vectorstore.similarity_search(user_question)

            llm=OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            
            st.write(response)
            
        

if __name__ == '__main__':
    main()