import streamlit as st
import streamlit_toggle as tog
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, OpenAI
import openai
# openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_documents_itr(docs):
    return iter(docs)

def get_text_chunks(text, is_doc_itr=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len
    )
    if not is_doc_itr:
        chunks = text_splitter.split_text(text)
    else:
        chunks = text_splitter.split_documents(text)

    return chunks


def get_vectorstore(text_chunks, is_doc_itr=False, use_openai=True):
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl",
            # model_name="intfloat/multilingual-e5-large",
            model_kwargs ={"device": "cpu"},
        )

    if not is_doc_itr:
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    else:
        vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, use_openai=True):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # if use_openai:
    #     llm = ChatOpenAI(temperature=0.2)
    # else:
    #     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.9, "max_length":512 })


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

def handle_system_input(user_question):
    response = st.session_state.conversation({'query': user_question})
    evalution = response['chat_history'][-1]
    st.write(bot_template.replace(
        "{{MSG}}", evalution.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    PARSE_AS_TEXT = True # False

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "target_doc_text" not in st.session_state:
        st.session_state.target_doc_text = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    USE_OPRNAI = True
    # USE_OPRNAI = tog.st_toggle_switch(label="**[experimental]** use OpenAI embeddings(set at the start of the conversation) or use HuggingFace's Instructor-XL embedding",
    #                 key="Key1",
    #                 default_value=True,
    #                 label_after = True,
    #                 inactive_color = '#D3D3D3',
    #                 active_color="#11567f",
    #                 track_color="#29B5E8"
    #                 )

    instruction = "given all the documents except TARGET_DOCUMENT, compare those documents with the TARGET_DOCUMENT and show the top 2 documents that are similar to the TARGET_DOCUMENT and explain your reasoning as well."
    instruction += "\nTARGET_DOCUMENT:\n"
    if user_question:
        handle_userinput(user_question)
    if "target_doc_text" in st.session_state and st.session_state.target_doc_text and len(st.session_state.target_doc_text) > 0:
        print(st.session_state.target_doc_text)
        handle_system_input(instruction+st.session_state.target_doc_text)
        st.session_state.target_doc_text = ""

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type = [".pdf"],
        )
        if pdf_docs:
            pdf_docs_to_compare = st.file_uploader(
                "Upload your target PDF that you want to be matched with the above pdfs, here and click on 'Process'", accept_multiple_files=False, type = [".pdf"],
            )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                # get pdf text
                # get_documents_itr
                raw_text = get_pdf_text(pdf_docs) if PARSE_AS_TEXT else get_documents_itr(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text, is_doc_itr=(not PARSE_AS_TEXT))

                # create vector store
                vectorstore = get_vectorstore(text_chunks, is_doc_itr=(not PARSE_AS_TEXT), use_openai=USE_OPRNAI)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, use_openai=USE_OPRNAI)

                if pdf_docs_to_compare and len(raw_text) != 0:
                    # get pdf text
                    st.session_state.target_doc_text = get_pdf_text([pdf_docs_to_compare]) if PARSE_AS_TEXT else get_documents_itr([pdf_docs_to_compare])


if __name__ == '__main__':
    main()
