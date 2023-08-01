from dotenv import load_dotenv
import requests
import pandas as pd
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template
import streamlit as st
import streamlit_toggle as tog
import torch

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import OpenAI
# from langchain.llms import HuggingFaceHub
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
# from transformers import AutoTokenizer, AutoModel
from huggingface_hub import InferenceClient
from huggingface_hub.inference_api import InferenceApi
import openai
# openai.api_key = st.secrets["OPENAI_API_KEY"]

# EMBEDDING_MODEL_ID = "hkunlp/instructor-xl"
EMBEDDING_MODEL_ID = "thenlper/gte-large"

def query(texts):
    API_URL = api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_ID}"
    # HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

def get_embeddings(texts):
    output = query(texts)
    embeddings = pd.DataFrame(output)
    print(embeddings)
    return embeddings

def get_pdf_text(pdf_docs):
    text = "here's a document:\n"
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        text+="\nhere's another document:\n"

    text = text[:-26]
    return text

def get_documents_itr(docs):
    return iter(docs)

def get_text_chunks(text, is_doc_itr=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    if not is_doc_itr:
        chunks = text_splitter.split_text(text)
    else:
        chunks = text_splitter.split_documents(text)

    return chunks


def get_vectorstore(text_chunks, is_doc_itr=False, use_openai=True):
    embeddings = get_embeddings(text_chunks)

    if not is_doc_itr:
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    else:
        vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, use_openai=True):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    llm = ChatOpenAI(temperature=0.2)
    # if use_openai:
    #     llm = ChatOpenAI(temperature=0.2)
    # else:
    #     llm = HuggingFaceHub(repo_id="stabilityai/StableBeluga2", model_kwargs={"temperature":0.2, "max_length":512 })


    # conversation_chain = ConversationalRetrievalChain.from_llm(
    conversation_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'query': user_question})
    st.session_state.chat_history = response['chat_history']
    chat_history = st.session_state.chat_history
    chat_history.reverse()

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    PARSE_AS_TEXT = True # False

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    USE_OPRNAI = False
    # USE_OPRNAI = tog.st_toggle_switch(label="**[experimental]** use OpenAI embeddings(set at the start of the conversation) or use HuggingFace's Instructor-XL embedding",
    #                 key="Key1",
    #                 default_value=True,
    #                 label_after = True,
    #                 inactive_color = '#D3D3D3',
    #                 active_color="#11567f",
    #                 track_color="#29B5E8"
    #                 )

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type = [".pdf"],
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                # get_documents_itr
                raw_text = get_pdf_text(pdf_docs) if PARSE_AS_TEXT else get_documents_itr(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text, is_doc_itr=(not PARSE_AS_TEXT))
                # st.session_state.text_chunks = get_text_chunks(raw_text, is_doc_itr=(not PARSE_AS_TEXT))

                # create vector store
                vectorstore = get_vectorstore(text_chunks, is_doc_itr=(not PARSE_AS_TEXT), use_openai=USE_OPRNAI)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, use_openai=USE_OPRNAI)


if __name__ == '__main__':
    main()
