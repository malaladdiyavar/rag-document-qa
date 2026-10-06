import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# 1. Connect to ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="llm_documents"
)


# ==========================================
# 2. Load stored documents
# ==========================================

data = collection.get(
    include=["documents", "metadatas"]
)

documents = data["documents"]


# ==========================================
# 3. Create TF-IDF vectorizer
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english"
)

document_vectors = vectorizer.fit_transform(
    documents
)


# ==========================================
# 4. Retrieve relevant documents
# ==========================================

def retrieve_documents(query, top_k=3):

    query_vector = vectorizer.transform(
        [query]
    )

    results = collection.query(
        query_embeddings=query_vector.toarray().tolist(),
        n_results=top_k
    )

    return results


# ==========================================
# 5. Generate grounded answer
# ==========================================

def generate_rag_answer(question, results):

    retrieved_documents = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    answer_parts = []

    answer_parts.append(
        "Based on the provided documents:\n"
    )

    for i, document in enumerate(
        retrieved_documents
    ):

        source = retrieved_metadata[i]["source"]

        # Take the most relevant part of the chunk
        words = document.split()

        if len(words) > 150:
            relevant_text = " ".join(words[:150])
        else:
            relevant_text = document

        answer_parts.append(
            f"\n[{i + 1}] Source: {source}\n"
            f"{relevant_text}"
        )

    return "\n".join(answer_parts)


# ==========================================
# 6. Main program
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    print(
        "\nRetrieving relevant documents..."
    )

    results = retrieve_documents(
        question,
        top_k=3
    )

    print(
        "\nGenerating grounded answer..."
    )

    answer = generate_rag_answer(
        question,
        results
    )

    print(
        "\n==================================="
    )

    print(
        "RAG ANSWER"
    )

    print(
        "==================================="
    )

    print(answer)

    print(
        "\n==================================="
    )

    print(
        "RETRIEVED SOURCES"
    )

    print(
        "==================================="
    )

    for i, metadata in enumerate(
        results["metadatas"][0]
    ):

        print(
            f"{i + 1}. {metadata['source']}"
        )