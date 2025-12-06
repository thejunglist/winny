# Winny

## Introduction
Due to the interest across the world in LLMs I decided to build my own chatbot that could be used to query documentation.
This project was built of curiosity and to test the limits of running models locally, rather than making calls through an API to a remote server.
I learned alot about computational power and how Large Language Models function under the hood.

The rough idea of the system is as follows:

1. A PDF document is loaded. In this case I chose an instruction manual.
2. The System then splits it into chunks
3. The chunks are then converted into embeddings and stored in a vector database.
4. These chunks are then retrieved and used to answer questions using an LLM.

For embedings I used OllamaEmbeddings, which uses Local Ollama models to convert text to numerical vectors.
I used Chroma as a database to store the vectors and search the embeddings.

# The Method
To load documents, I used the below function:
`def load_documents():
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()`
This function finds the PDF file at data/kramer_vp_778_um_rev_7.pdf. The document is then loaded page by page, and then returns a list where each item contains the text from one page.

The documents are then split using:
`def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(documents)`

  LLMs have limited context windows, or in other words how much information they can reliably retreive, and smaller chunks improve this action.

After this the embeddings are converted from text into numerical vectors

`def get_embedding_function(model_name="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model_name)`

This uses the nomic-embed-text model running locally via ollama. As the name suggests, this model can only be used to embed text.

The indexing of the documents is done like so below:
`def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )`
The function above takes all chunks of text and converts them to a vector(embedding). These vectors are stored in a chroma database. An index is then created for fast similarity search. A similarity search is a way of searching for multimedia files that are similar, rather than traditional search methods.

The Retrevial Augemented Generation chain is created in the function below:
`def create_rag_chain(vector_store, llm_model_name="qwen3:8b"):
    llm = ChatOllama(model=llm_model_name, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})`
This is where I defined the model, which i decided to be qwen3:8b because of its relatively small size and decent enough rating for these kinds of tasks.
The temperature in this case is 0 which means that the responses that it gives are deterministic and factual.

As a prompt template I used:
`template = """Answer the question based ONLY on the following context: {context}
Question: {question}
"""`
This tells the LLM only to use retrieved info, not its general knowledge.

The LangChain Expression language is a syntax created by langchain to create chains.
`rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
)`
The code above expression has 4 parts:
1. Input user's question
2. Retriever: Searches the chroma db to find the three most relevant chunks
3. Prompt fills template with context + question
4. LLM generates answer based on the prompt
5. Parser: Extracts plain text from LLM response

The querying, which sends the question through the entire chain and returns the LLM's answer based on retrieved context:

`def query_rag(chain, question):
    response = chain.invoke(question)`

    
The main excecution of the program is done like this:
`if __name__ == "__main__":
    docs = load_documents()              # Load PDF
    chunks = split_documents(docs)       # Split into chunks
    embedding_function = get_embedding_function()  # Setup embeddings
    vector_store = index_documents(chunks, embedding_function)  # Index
    rag_chain = create_rag_chain(vector_store)  # Create chain
    
    query_rag(rag_chain, "What is the topic of this document")
    query_rag(rag_chain, "what is the function of this device")`

## Flow
The flow of the rag is: 
1.Process the PDF once
2.Create searchable database
3.Ask questions that search the database and use the LLM to answer.


