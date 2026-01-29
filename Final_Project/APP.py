import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (PyPDFLoader,TextLoader,CSVLoader, WebBaseLoader,DirectoryLoader,Docx2txtLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import nest_asyncio
from llama_parse import LlamaParse
from langchain_core.documents import Document

nest_asyncio.apply()
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
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()



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
        [".PDF/CSV/TXT/DOCX", "Directory", "WEB-URL"]
    )
  


    if selected == ".PDF/CSV/TXT/DOCX":
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "csv", "txt","docx"],
            accept_multiple_files=True,
        )

    elif selected == "Directory":
        dir_path = st.text_input("Enter directory path")

    elif selected == "WEB-URL":
        url = st.text_input("Enter website URL")
    # Place this inside `with st.sidebar:`
    
    if st.button("🗑️ Reset Brain / Clear History"):
        st.session_state.messages = []
        st.session_state.processed_files = set()          
        st.rerun() # Refreshes the app
 

# -------------------- HELPERS -------------------
def create_temp_paths(files):
    file_data = []
    for file in files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.name).suffix
        ) as tmp:
            tmp.write(file.getbuffer())
        file_data.append((tmp.name, file.name))
    return file_data


def load_files(file_data):
    docs = []
    for path,original_name in file_data:
        suffix = Path(path).suffix.lower()
        loader = None
        try:
            if suffix == ".pdf":
                parser = LlamaParse(
                    result_type="markdown",
                    verbose=True,
                    language="en",
                    system_prompt="Extract all text. If you see a graph, chart, or image, describe it in detail textually."
                )
                llama_docs = parser.load_data(path)
                for doc in llama_docs:
                    docs.append(Document(
                        page_content=doc.text,
                        metadata={"source": original_name}
                    ))

            else:
                if suffix == ".txt":
                    loader = TextLoader(path, encoding="utf-8")
                elif suffix == ".csv":
                    loader = CSVLoader(path)
                elif suffix == ".docx":
                    loader = Docx2txtLoader(path)
            if loader:
                temp_docs = loader.load()
                for d in temp_docs:
                    d.metadata["source"] = original_name # <--- Fix for other files
                docs.extend(temp_docs)            
            else:
                continue

        except Exception as e:
            st.error(f"Error loading {original_name}: {e}")
            
    return docs


def load_directory(dir_path):
    pdf = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt = DirectoryLoader(
        dir_path, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    csv = DirectoryLoader(dir_path, glob="**/*.csv", loader_cls=CSVLoader)
    return pdf.load() + txt.load() + csv.load()


import time # <--- Make sure to import this at the top if missing

def get_embeddings():
    
    return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
   

def get_vector_store(chunks, collection_name):
    embeddings = get_embeddings()

    store = Chroma(
        persist_directory=VECTOR_DB_DIR,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    if chunks: 
        batch_size = 20  
        total_chunks = len(chunks)
        
        progress_text = "⏳ Embedding documents... Please wait."
        my_bar = st.sidebar.progress(0, text=progress_text)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            try:
                store.add_documents(batch)
            except Exception as e:
                if "429" in str(e):
                    
                    time.sleep(20) 
                    store.add_documents(batch) 
                else:
                    raise e 
            progress = min((i + batch_size) / total_chunks, 1.0)
            my_bar.progress(progress, text=f"⏳ Embedded {int(progress*100)}% of data...")
            
            time.sleep(2)
        my_bar.empty()
        st.toast(f"✅ Successfully added {total_chunks} chunks!")

    return store
def format_chat_history(messages, max_turns=6):
    """
    Convert chat history into a single string.
    Keeps last N turns only (token safe).
    """
    history = []
    for msg in messages[-max_turns * 2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {msg['content']}")
    return "\n".join(history)


# -------------------- LOAD DATA --------------------
documents = []

if selected == ".PDF/CSV/TXT/DOCX" and uploaded_files:
    file_data = create_temp_paths(uploaded_files)
    for path, original_name in file_data:
        if original_name not in st.session_state.processed_files:
            docs = load_files([(path, original_name)])
            documents.extend(docs)
            st.session_state.processed_files.add(original_name)
        
        os.remove(path)  
elif selected == "Directory" and dir_path and os.path.isdir(dir_path):
    st.session_state.processed = False
    documents = load_directory(dir_path)

elif selected == "WEB" and url:
    st.session_state.processed = False
    documents = WebBaseLoader([url]).load()

with st.sidebar:
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
    input_variables=["context", "chat_history", "question"],
    template="""
You are a helpful assistant.

Use the chat history ONLY for conversational continuity.
Use the document context as the source of truth.

Chat History:
{chat_history}

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
st.markdown(
    """
    <style>
    /* Target the chat input container */
    div[data-testid="stChatInput"] {
        border: 2px solid cyan !important; /* Green Border */
        border-radius: 10px;
        background-color: #2E2E2E;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
query = st.chat_input("Ask a question about your documents")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    chat_history = format_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(query)

    retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3})


    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    with st.spinner("Thinking..."):
        answer = chain.invoke(
            {"context": context,"chat_history": chat_history,"question": query,}
        )

    with st.chat_message("assistant"):
        st.markdown(answer)
        unique_sources = {doc.metadata.get('source', 'Unknown') for doc in docs}
        
        with st.expander(f"📚 References ({len(unique_sources)} files used)"):
            for source in unique_sources:
                    # Clean up the path to show just the filename
                filename = Path(source).name 
                st.write(f"📄 **{filename}**")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

