import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader,DirectoryLoader
from pathlib import Path
import tempfile
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
import os

# --- Constants ---
VECTOR_DB_DIR = "vector_db"
COLLECTION_DOCS = "docs_collection"
COLLECTION_WEB = "web_collection"

# --- Initialization ---
load_dotenv()
st.set_page_config(page_icon="🧠", page_title="The Alternate Brain", initial_sidebar_state="expanded", layout="wide")

st.title("The Alternate Brain")
st.caption("Your Private AI Friend")
st.info("You can upload multiple documents and can ask any questions the model will give you answer.")
with st.chat_message("assistant"):
    st.markdown("Welcome ! How can i help you ?")

uploaded_files = None
url = None
with st.sidebar:
    st.header("Upload Here Please ↓")
    selected = st.radio("Select the type you want to:", [".PDF.CSV.TXT","Directory", "WEB-URL"])

    if selected == ".PDF.CSV.TXT":
        uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "csv", "txt"], accept_multiple_files=True)
    
    elif selected == "Directory":
        dir_path = st.text_input("Enter the directory path (Ex: C:/Users/Docs)")
    
    elif selected == "WEB-URL":
        url = st.text_input("Enter the site you want to get the information (Ex: https://example.com)")



# --- Session State for URLs ---
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = set()

if "processed_data" not in st.session_state:
    st.session_state.processed_data=set()

if selected == "WEB-URL" and url:
    if url in st.session_state.processed_urls:
        with st.sidebar:
            st.info("URL already processed. Using existing vectors.")
    else:
        st.session_state.processed_urls.add(url)

# --- Helper Functions ---
def path_creator(uploaded_files):
    if not uploaded_files:
        temp_path=[]

        return []
    for file in uploaded_files:
        # Create a temp file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
            tmp.write(file.getbuffer())
            temp_path.append(tmp.name)
    return temp_path

def loaders(paths):
    documents = []
    for path in paths:
        suffix = Path(path).suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(path)
            elif suffix == ".txt":
                loader = TextLoader(path, encoding="utf-8")
            elif suffix == ".csv":
                loader = CSVLoader(path)
            else:
                st.warning(f"Unsupported file format: {suffix}")
                continue
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
    return documents



def dir_loader(dir_path):
    documents=[]
    if selected == "Directory" and dir_path:
        if os.path.isdir(dir_path):
            st.info(f"Scanning directory: {dir_path}")
            try:
            # We run 3 loaders to capture different file types in that folder
            # glob="**/*.pdf" means search recursively in all subfolders too
                pdf_loader = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
                txt_loader = DirectoryLoader(dir_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
                csv_loader = DirectoryLoader(dir_path, glob="**/*.csv", loader_cls=CSVLoader)
            
            # Combine results from all 3 scans
                documents = pdf_loader.load() + txt_loader.load() + csv_loader.load()
            
                if not documents:
                    st.warning("No valid files (PDF, TXT, CSV) found in that folder.")
                else:
                    with st.sidebar:
                        st.success(f"Found {len(documents)} documents in directory.")
            except Exception as e:
                st.error(f"Error loading directory: {e}")
        else:
            st.error("Invalid directory path. Please check the path and try again.")
        return documents

def embedding_Storing(chunks, collection_name):
   
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # Initialize Chroma (loads existing if present, creates new if not)
    vector_store = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    # If we have new chunks to process, add them to the DB
    if chunks:
        vector_store.add_documents(documents=chunks)
        # Note: Recent versions of Chroma persist automatically, but this ensures safety
        # vector_store.persist() 
        st.toast(f"Added {len(chunks)} chunks to {collection_name}")

    return vector_store

# --- Main Logic ---

documents = []
paths = []

# 1. Load Documents
if selected == ".PDF.CSV.TXT" and uploaded_files:
    paths = path_creator(uploaded_files)
    documents = loaders(paths)
elif selected == "WEB-URL" and url:
    # Only load if not already processed to avoid duplicates (optional logic)
    # For now, we load it so we can split it.
    try:
        web_loader = WebBaseLoader(web_paths=[url])
        documents = web_loader.load()
    except Exception as e:
        st.error(f"Error loading URL: {e}")
# Clean up temp files immediately after loading
    for path in paths:
        try:
            os.remove(path)
        except:
            pass
elif selected == "Directory" and dir_path:
    if dir_path not in st.session_state.processed_data:
        documents = dir_loader(dir_path)
        if documents:
            st.session_state.processed_data.add(dir_path)
    else:
        st.sidebar.info(f"Directory '{dir_path}' already processed.")
# 2. Check if we need to stop (no docs and no DB)
if not documents and not os.path.exists(VECTOR_DB_DIR):
    st.info("Upload documents or provide a URL to continue.")
    # We stop strictly only if we have NOTHING to work with
    if not uploaded_files and not url:
        st.stop()

# 3. Split Text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents) if documents else []

# 4. Determine Collection Name (Fixing the NameError)
current_collection_name = COLLECTION_WEB if selected == "WEB-URL" else COLLECTION_DOCS

# 5. Store/Retrieve Embeddings
# We pass the collection name explicitly now
store = embedding_Storing(chunks, current_collection_name)
print("Chunks created:", len(chunks))

if os.path.exists(VECTOR_DB_DIR) and not chunks:
    with st.sidebar:
        st.success("Vector database loaded from disk.")
elif chunks:
    st.success("Vector database updated with new documents.")


# --- Chat Chain Setup ---

prompt_template = PromptTemplate(
    input_variables=["history","context", "question"],
    template="""You are a helpful assistant.
    
    Instruction: 
    1. First, search the "Context" below for the answer.Use the "Chat History" and "Context" to answer the "Question".
    2. Use the context to provide detailed answer to the user.
    3. If the answer is NOT in the Context, you must answer the user's "Question" using your own general knowledge, but you must start the response with "From the Internet:".
    4. Ensure your answer is directly relevant to the specific "Question" asked.
    
    Chat History (Summary):
    {history}

    Context:
    {context}

    Question:
    {question}

    Answer:"""
)

# Note: Ensure "gemini-2.5-flash" is a valid model name for your API key. 
# Usually, it is "gemini-1.5-flash" or "gemini-pro".
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0) 

parser = StrOutputParser()
chain = prompt_template | llm | parser

# --- Chat Interface ---
if "memory" not in st.session_state:
    # We use the LLM to summarize older messages
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,  
        memory_key="history",
        return_messages=False  
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Query
query = st.chat_input("Ask a question about your documents:")

if query:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    memory_variables = st.session_state.memory.load_memory_variables({})
    history_text = memory_variables['history']
    # Retrieve context
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.8})
    contexts = retriever.invoke(query)
    
    # Prepare context text
    context_text = "\n\n".join([doc.page_content for doc in contexts])
    
    # Generate Answer
    with st.spinner("Thinking..."):
        response = chain.invoke({"history": history_text,"context": context_text, "question": query})
    st.session_state.memory.save_context({"input": query}, {"output": response})
    # Display and Save Assistant Response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})