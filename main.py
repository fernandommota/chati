


from get_documents import read_csv_data

from functions import get_chroma_client, generate_embeddings_and_persist_chroma_db, run_mini_check

#ids, documents, metadatas = read_csv_data()

#print(ids, documents, metadatas)

chroma_client = get_chroma_client()

#generate_embeddings_and_persist_chroma_db(chroma_client, ids, documents, metadatas)

# Query the vector database
def query_vector_database(query_texts):
    collection = chroma_client.get_or_create_collection(name="chati")

    # Query mispelled word: 'vermiceli'. Expect to find the correctly spelled 'vermicelli' item
    results = collection.query(
        query_texts=query_texts,
        n_results=5,
        include=['documents', 'distances', 'metadatas']
    )

    # ids, embeddings, documents, uris, data, metadatas, distances, included
    for index, result in enumerate(results['ids']):
        print(result)
        print(results['distances'][index])
    
    for index, result in enumerate(results['documents']):
        print(index)
        print('documents', results['documents'][index])
        run_mini_check(query_texts[0], results['documents'][index])
        
query_texts = ["Who is Alpine Hardwood (Aust)?"]
query_vector_database(query_texts)