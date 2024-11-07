

from get_documents import read_csv_data

from functions import get_chroma_client, generate_embeddings_and_persist_chroma_db

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
        #for document in results['documents']:
        #    print(document)
        
query_texts = ["Hexal Australia Pty Ltd v Roche Therapeutics Inc (2005) 66 IPR 325, the likelihood of irreparable harm was regarded by Stone J as, indeed, a separate element that had to be established by an applicant for an interlocutory injunction."]
query_vector_database(query_texts)

