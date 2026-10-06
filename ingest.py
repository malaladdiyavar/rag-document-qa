import os
import chromadb
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# 1. Extract text from PDF
# ==========================================

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================
# 2. Split text into chunks
# ==========================================

def chunk_text(text, chunk_size=600, overlap=100):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# 3. Read all PDF documents
# ==========================================

def load_documents():

    documents_folder = "./documents"

    all_chunks = []
    all_ids = []
    all_metadatas = []

    document_files = [
        file
        for file in os.listdir(documents_folder)
        if file.lower().endswith(".pdf")
    ]

    print(f"Found {len(document_files)} PDF files.")

    chunk_id = 0

    for filename in document_files:

        pdf_path = os.path.join(
            documents_folder,
            filename
        )

        print(f"\nProcessing: {filename}")

        text = extract_text_from_pdf(pdf_path)

        print(
            f"Extracted characters: {len(text)}"
        )

        chunks = chunk_text(
            text,
            chunk_size=600,
            overlap=100
        )

        print(
            f"Created chunks: {len(chunks)}"
        )

        for chunk in chunks:

            all_chunks.append(chunk)

            all_ids.append(
                f"chunk_{chunk_id}"
            )

            all_metadatas.append({
                "source": filename,
                "chunk_number": chunk_id
            })

            chunk_id += 1

    return (
        all_chunks,
        all_ids,
        all_metadatas
    )


# ==========================================
# 4. Main ingestion function
# ==========================================

def ingest_documents():

    # Load document chunks
    (
        all_chunks,
        all_ids,
        all_metadatas
    ) = load_documents()

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    # ======================================
    # Create TF-IDF embeddings
    # ======================================

    print(
        "\nCreating local TF-IDF embeddings..."
    )

    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english"
    )

    embeddings = vectorizer.fit_transform(
        all_chunks
    ).toarray()

    print(
        f"Embedding dimension: {embeddings.shape[1]}"
    )

    # ======================================
    # Create ChromaDB
    # ======================================

    print(
        "\nCreating ChromaDB collection..."
    )

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    # Delete old collection if it exists
    try:
        chroma_client.delete_collection(
            name="llm_documents"
        )
        print("Old collection deleted.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="llm_documents"
    )

    # ======================================
    # Store embeddings in ChromaDB
    # ======================================

    print(
        "\nStoring embeddings in ChromaDB..."
    )

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas
    )

    # ======================================
    # Verify
    # ======================================

    stored_count = collection.count()

    print("\n===================================")
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("===================================")

    print(
        f"Total chunks created : {len(all_chunks)}"
    )

    print(
        f"Total chunks stored  : {stored_count}"
    )


# ==========================================
# 5. Run program
# ==========================================

if __name__ == "__main__":
    ingest_documents()