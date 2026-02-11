# Setting up User Interface for the Chatbot through Streamlit for phase 1
import streamlit as st
from dotenv import load_dotenv
import os

#LLM Setup Dependencies

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings  # Modern HF integration
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
st.title("RAG Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_path = "" 
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found")
        return None
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

prompt = st.chat_input("Ask something from your PDF...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    try:
       vectorstore = get_vectorstore()
       if vectorstore:

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )

        qa_system_prompt = """Answer the question based only on the following context:
            <context>
            {context}
            </context>"""
            
        qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                ("human", "{input}"),
            ])

        document_chain = create_stuff_documents_chain(llm, qa_prompt)
            
        retrieval_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 3}), 
                document_chain
            )

        result = retrieval_chain.invoke({"input": prompt})
        response = result["answer"]

        with st.chat_message("assistant"):
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")