# quickrag.py

import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="QuickRAG", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f9fafb;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 8px;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
        }
        .stButton > button {
            border-radius: 10px;
            background-color: #4F46E5;
            color: white;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #4338CA;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Title ----------
st.title("⚡ QuickRAG")
st.caption("Your friendly document & web page Q&A assistant, powered by Groq + LangChain")

# ---------- API Key ----------
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("🚨 GROQ_API_KEY not found in Streamlit secrets!")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_api_key

# ---------- Function to process documents ----------
def process_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    splits = text_splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model="llama-3.1-8b-instant")

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ---------- Document Source Selection ----------
with st.expander("📥 Upload or Link Your Data", expanded=True):
    mode = st.radio("Choose document source:", ["PDF", "Web URL"], horizontal=True)
    docs = None
    rag_chain = None

    if mode == "Web URL":
        link = st.text_input("🌐 Enter webpage URL:")
        if st.button("Load Web Data", use_container_width=True):
            if link:
                with st.spinner("🔍 Fetching and processing webpage..."):
                    loader = WebBaseLoader(web_paths=[link])
                    docs = loader.load()
                    rag_chain = process_documents(docs)
                st.success("✅ Web data loaded successfully!")
            else:
                st.warning("Please enter a valid URL.")

    elif mode == "PDF":
        pdf_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
        if st.button("Load PDF Data", use_container_width=True):
            if pdf_file:
                with st.spinner("📚 Processing PDF..."):
                    loader = PyPDFLoader(pdf_file)
                    docs = loader.load()
                    rag_chain = process_documents(docs)
                st.success("✅ PDF data loaded successfully!")
            else:
                st.warning("Please upload a PDF first.")

# ---------- Chat Interface ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if rag_chain:
    st.subheader("💬 Chat with QuickRAG")
    user_input = st.chat_input("Ask anything about your document...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🤔 Thinking..."):
            answer = rag_chain.invoke(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    st.info("⬆️ Load a PDF or webpage above to start chatting.")
