import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (PyPDFLoader,TextLoader,CSVLoader, WebBaseLoader,DirectoryLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------- CONSTANTS --------------------
VECTOR_DB_DIR = "vector_db"
COLLECTION_DOCS = "docs_collection"
COLLECTION_WEB = "web_collection"


# -------------------- INIT --------------------
load_dotenv()

st.set_page_config(
    page_icon="🧠",
    page_title="The Alternate Brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("The Alternate Brain")
st.caption("Your Private AI Friend")
st.info("Upload documents or a URL and ask questions based on the content.")

with st.chat_message("assistant"):
    st.markdown("Welcome! How can I help you?")


# -------------------- SIDEBAR --------------------
uploaded_files = None
url = None
dir_path = None

with st.sidebar:
    st.header("Upload Here ↓")
    selected = st.radio(
        "Select input type:",
        [".PDF / CSV / TXT", "Directory", "WEB-URL"]
    )

    if selected == ".PDF / CSV / TXT":
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "csv", "txt"],
            accept_multiple_files=True,
        )

    elif selected == "Directory":
        dir_path = st.text_input("Enter directory path")

    elif selected == "WEB-URL":
        url = st.text_input("Enter website URL")


# -------------------- HELPERS --------------------
def create_temp_paths(files):
    paths = []
    for file in files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.name).suffix
        ) as tmp:
            tmp.write(file.getbuffer())
            paths.append(tmp.name)
    return paths


def load_files(paths):
    docs = []
    for path in paths:
        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(path)
        elif suffix == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        elif suffix == ".csv":
            loader = CSVLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs


def load_directory(dir_path):
    pdf = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt = DirectoryLoader(
        dir_path, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    csv = DirectoryLoader(dir_path, glob="**/*.csv", loader_cls=CSVLoader)
    return pdf.load() + txt.load() + csv.load()


def get_vector_store(chunks, collection_name):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    store = Chroma(
        persist_directory=VECTOR_DB_DIR,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    if chunks:
        store.add_documents(chunks)
        st.toast(f"Added {len(chunks)} chunks")

    return store


# -------------------- LOAD DATA --------------------
documents = []

if selected == ".PDF / CSV / TXT" and uploaded_files:
    paths = create_temp_paths(uploaded_files)
    documents = load_files(paths)
    for p in paths:
        os.remove(p)

elif selected == "Directory" and dir_path and os.path.isdir(dir_path):
    documents = load_directory(dir_path)

elif selected == "WEB-URL" and url:
    documents = WebBaseLoader([url]).load()

if not documents and not os.path.exists(VECTOR_DB_DIR):
    st.info("Please upload documents or provide a URL.")
    st.stop()


# -------------------- SPLIT + STORE --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents) if documents else []

collection = COLLECTION_WEB if selected == "WEB-URL" else COLLECTION_DOCS
store = get_vector_store(chunks, collection)


# -------------------- LLM + PROMPT --------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Use the Context below to answer the Question.
If the answer is not in the Context, answer from general knowledge
and start with "From the Internet:".

Context:
{context}

Question:
{question}

Answer:
""",
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

chain = prompt | llm | StrOutputParser()


# -------------------- CHAT --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your documents")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):
        st.markdown(query)

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.8},
    )

    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    with st.spinner("Thinking..."):
        answer = chain.invoke(
            {"context": context, "question": query}
        )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
