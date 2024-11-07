
import ollama
import warnings
import nltk
import chromadb
from chromadb.utils import embedding_functions

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)
nltk.download("punkt_tab", quiet=True)

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

def check_document_claim(document, claim):
    """Checks if the claim is supported by the document by calling bespoke-minicheck.

    Returns Yes/yes if the claim is supported by the document, No/no otherwise.
    Support for logits will be added in the future.

    bespoke-minicheck's system prompt is defined as:
      'Determine whether the provided claim is consistent with the corresponding
      document. Consistency in this context implies that all information presented in the claim
      is substantiated by the document. If not, it should be considered inconsistent. Please
      assess the claim's consistency with the document by responding with either "Yes" or "No".'

    bespoke-minicheck's user prompt is defined as:
      "Document: {document}\nClaim: {claim}"
    """
    prompt = f"Document: {document}\nClaim: {claim}"
    response = ollama.generate(
        model="bespoke-minicheck", prompt=prompt, options={"num_predict": 2, "temperature": 0.0}
    )

    return response["response"].strip()


def run_mini_check(question, documents):
    
    sourcetext = "\n\n".join(documents)

    print(f"\nRetrieved chunks: \n{sourcetext}\n")

    # Give the retreived chunks and question to the chat model
    system_prompt = f"Only use the following information to answer the question. Do not use anything else: {sourcetext}"

    ollama_response = ollama.generate(
        model="llama3.2",
        prompt=question,
        system=system_prompt,
        options={"stream": False},
    )

    answer = ollama_response["response"]
    print(f"LLM Answer:\n{answer}\n")

    # Check each sentence in the response for grounded factuality
    if answer:
        supported_claims_yes = []
        supported_claims_no = []
        for claim in nltk.sent_tokenize(answer):
            is_supported_claim = check_document_claim(sourcetext, claim)
            
            if is_supported_claim.strip() == "Yes":
                supported_claims_yes.append(claim)
            else:
                supported_claims_no.append(claim)
            print(f"LLM Claim: {claim}")
            print(
                f'Is this claim supported by the context according to bespoke-minicheck? {check_document_claim(sourcetext, claim)}'
            )
        
        print('\n\n Results Summary')
        print('Total: ',len(supported_claims_yes) + len(supported_claims_no))
        print('Yes: ',len(supported_claims_yes))
        print('No: ',len(supported_claims_no))