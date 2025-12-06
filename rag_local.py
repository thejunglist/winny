import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader 
from langchain_ollama import OllamaEmbeddings
load_dotenv() 
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
DATA_PATH = "data/"
PDF_FILENAME = "kramer_vp-778_um_rev_7.pdf"
CHROMA_PATH = "chroma_db"


def load_documents():
    """loads documents from specified path"""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    # loader = UnstructuredPDFLoader(pdf_path) #alternative
    documents = loader.load()
    print(f"loaded {len(documents)} pages(s) from {pdf_path}")
    return documents
# documents = load_documents() #call this later

def split_documents(documents):
    """splits documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"split into {len(all_splits)} chunks")
    return all_splits


def get_embedding_function(model_name="nomic-embed-text"):
    """initializes the ollama embedding function."""
    # make sure Ollama server is running (ollama serve)
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"initialized Ollama embeddings with model: {model_name}")
    return embeddings

#embedding_function = get_embedding_function() # call this later

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Initializes or loads the chroma vector store."""
    vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embedding_function)

    print(f"Vector store initialised/loaded from: {persist_directory}")
    return vectorstore


def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Indexes document chuncks into the Chroma vector store"""
    print(f"Indexing {len(chunks)} chunks...")
    # Use from_documents for initial creation
    # This will overwrite existing data if the directory exists but isnt a valid Chroma DB.
    vectorstore = Chroma.from_documents(
            documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectorstore.persist() # Ensure data is saved
    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore

def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    """creates the rag chain"""
    # initialize the LLM
    llm = ChatOllama(
        model=llm_model_name,temperature=0, # lower temp for more factual answers
        num_ctx=context_window # important: set context window size
    )
    print (f"Initialized ChatOllama with model: {llm_model_name}, context window: {context_window}")

    # create the retriever
    retriever = vector_store.as_retriever(
        search_type = "similarity", # or mmr
        search_kwargs={'k':3} # retrieve top 3 chunks
)
    print("Retriever initialised.")

# define the prompt template
    template = """Answer the question based ONLY on the following context: {context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("prompt template created.")

#Define the RAG chain using LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        |prompt
        |llm
        |StrOutputParser()
    )
    print("RAG chain created")
    return rag_chain

def query_rag(chain, question):
    """queries the rag chain and prints the response."""
    print("\nQuerying RAG chain...")
    print(f"question: {question}")
    response = chain.invoke(question)
    print("\nResponse:")
    print(response)

# --- Main Excecution ---
if __name__ == "__main__":
    #load docs
    docs = load_documents()

    #split docs
    chunks = split_documents(docs)

    #get embedding function
    embedding_function = get_embedding_function() # using ollama nomic text

    # index documents (only needs to be done once per doc set)
    #check if db exists, if not index.
    print("attempting to index documents...")
    vector_store = index_documents(chunks, embedding_function)
    #to load existing db instead:
    #vector_store = get_vector_store(embedding_function)

    #5 create rag chain
    rag_chain = create_rag_chain(vector_store, llm_model_name="qwen3:8b")

    #6 query

    query_question = "What is the topic of this document"
    query_rag(rag_chain, query_question)

    query_question_2 = "what is the function of this device"
    query_rag(rag_chain, query_question_2)

 
