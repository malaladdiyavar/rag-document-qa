import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# Connect to ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="llm_documents"
)


# ==========================================
# Load stored documents
# ==========================================

data = collection.get(
    include=["documents", "metadatas"]
)

documents = data["documents"]


# ==========================================
# Create TF-IDF vectorizer
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english"
)

vectorizer.fit(documents)


# ==========================================
# Retrieve relevant documents
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
# Answer WITH RAG
# ==========================================

def answer_with_rag(question):

    results = retrieve_documents(
        question,
        top_k=3
    )

    documents = results["documents"][0]
    sources = results["metadatas"][0]

    # Use the first retrieved document
    retrieved_text = documents[0]

    # Find the important Transformer statement
    keyword = (
        "In this work we propose the Transformer"
    )

    start = retrieved_text.find(keyword)

    if start != -1:

        answer = retrieved_text[start:]

        # Stop after a reasonable length
        if len(answer) > 700:
            answer = answer[:700]

    else:

        answer = retrieved_text[:700]

    return answer, sources


# ==========================================
# Answer WITHOUT RAG
# ==========================================

def answer_without_rag(question):

    return (
        "The Transformer is a neural network "
        "architecture based on attention mechanisms. "
        "It processes sequences without relying on "
        "recurrent networks."
    )


# ==========================================
# Main comparison
# ==========================================

if __name__ == "__main__":

    question = (
        "What is the Transformer architecture?"
    )

    print("\n===================================")
    print("RAG VS WITHOUT RAG COMPARISON")
    print("===================================")

    # --------------------------------------
    # Without RAG
    # --------------------------------------

    print("\nWITHOUT RAG")
    print("-----------------------------------")

    no_rag_answer = answer_without_rag(
        question
    )

    print(no_rag_answer)


    # --------------------------------------
    # With RAG
    # --------------------------------------

    print("\nWITH RAG")
    print("-----------------------------------")

    rag_answer, sources = answer_with_rag(
        question
    )

    print(rag_answer)


    # --------------------------------------
    # Sources
    # --------------------------------------

    print("\nSOURCES USED BY RAG")
    print("-----------------------------------")

    for i, metadata in enumerate(sources):

        print(
            f"{i + 1}. {metadata['source']}"
        )


    # --------------------------------------
    # Summary
    # --------------------------------------

    print("\n===================================")
    print("COMPARISON SUMMARY")
    print("===================================")

    print(
        "Without RAG: General knowledge-based answer."
    )

    print(
        "With RAG   : Answer based on retrieved "
        "content from the provided documents."
    )

    print(
        "==================================="
    )