from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import MWDumpLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from typing import List, Optional
from dotenv import load_dotenv
import os

def choose_option():
    option = -1;
    while option not in ["1", "2", "3"]:
        print("""What do you want to do?
    1. Skip indexing and work with the persisted data in data/chroma
    2. Index new data from data/comics.xml
    3. Delete the persisted data in data/chroma""")
        option = input("Option: ")
    return option

option = choose_option()
    
load_dotenv() 
api_key = os.getenv("GOOGLE_API_KEY")

# Credit: https://www.reddit.com/r/developersIndia/comments/1czsbmf/comment/l60fxtt/
# GoogleGenerativeAIEmbeddings returns a Repeated object, but ChromaDB expects a list, so we need to convert it
class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], task_type: Optional[str] = None, titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None) -> List[List[float]]:
        embeddings_repeated = super().embed_documents(texts, task_type, titles, output_dimensionality)
        # convert proto.marshal.collections.repeated.Repeated to list
        embeddings = [list(emb) for emb in embeddings_repeated]
        return embeddings

# Load the models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
embeddings = CustomGoogleGenerativeAIEmbeddings(model="models/text-embedding-004", )
vectordb = Chroma(embedding_function=embeddings, persist_directory="data/chroma")
if option == "3":
    vectordb.delete(vectordb.get()["ids"])
    exit()

elif option == "2":
    ## INDEXING ##
    # Load the documents
    filepath = "data/comics.xml"
    loader = MWDumpLoader(
        file_path=filepath,
        encoding="utf8",
        # namespaces = [0,2,3] Optional list to load only specific namespaces. Loads all namespaces by default.
        skip_redirects=True,  # will skip over pages that just redirect to other pages (or not if False)
        stop_on_error=False,  # will skip over pages that cause parsing errors (or not if False)
    )
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n", " "],
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = lambda text: len(text),
        is_separator_regex=False
    )
    documentsChunks = text_splitter.split_documents(documents)

# Transform the chunks into embeddings and store them in a vector database
    batchSize = 75
    totalDocumentsChunks = len(documentsChunks)
    for i in range(0, totalDocumentsChunks, batchSize):
        # There is a limit of 100 documents per call, so we split into smaller batches
        print(f"Embedding documents from {i} to {min(i+batchSize, totalDocumentsChunks)} out of a total of {totalDocumentsChunks}. ({i/totalDocumentsChunks*100:.2f}% done)")
        batch = documentsChunks[i:i+batchSize]
        vectordb.add_documents(batch)

## RETRIEVAL AND GENERATION ##
# Use Chroma as a retriever, number of documents to be retrieved = 8
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# Template for the llm
prompt = PromptTemplate.from_template("""
    You are a helpful AI assistant.
    Respond based on the provided context.
    context: {context}
    input: {input}
    answer:""")

# Receives a list of documents and formats all of them into the prompt
documentProcessingChain = create_stuff_documents_chain(llm, prompt)

# Receives an input, which is then passed to the retriever to fetch relevant documents
documentRetrievalChain = create_retrieval_chain(retriever, documentProcessingChain)

# Invoke the retrieval chain
querie = input("Digite sua pergunta: ")
response = documentRetrievalChain.invoke({"input":querie})

# Print the response
def print_formatted(response):
    print("Input:", response['input'])
    print("\nRetrieved Documents:")
    for i, doc in enumerate(response['context']):
        print(f"\nDocument {i+1}:")
        print(f"Page Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-"*50)
    print("\nAnswer:", response['answer'])

print_formatted(response)

