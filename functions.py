import chromadb
from chromadb.utils import embedding_functions

def get_chroma_client():
    # Reference: https://docs.trychroma.com/getting-started
    # Instantiate chromadb instance. Data is stored in memory only.
    # chroma_client = chromadb.Client()
    # Instantiate chromadb instance. Data is stored on disk (a folder named 'my_vectordb' will be created in the same folder as this file).
   

    return chromadb.PersistentClient(path="chati")

def generate_embeddings_and_persist_chroma_db(chroma_client, ids, documents, metadatas):
    
    # Select the embedding model to use.
    # List of model names can be found here https://www.sbert.net/docs/pretrained_models.html
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Use this to delete the database
    chroma_client.delete_collection(name="chati")

    # Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
    collection = chroma_client.get_or_create_collection(name="chati", embedding_function=sentence_transformer_ef)

    # Add all the data to the vector database. ChromaDB automatically converts and stores the text as vector embeddings. This may take a few minutes.
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )