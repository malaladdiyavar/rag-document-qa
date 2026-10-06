\# RAG Document Question Answering System



\## 1. Project Overview



This project implements a Retrieval-Augmented Generation (RAG) based document question-answering system.



The system allows users to ask questions about a collection of research papers. Instead of depending only on general model knowledge, the system searches the provided documents, retrieves relevant text chunks, and uses those retrieved documents as the basis for answering the question.



The project demonstrates how document retrieval can be combined with a language-model-style question-answering workflow to provide answers grounded in a user's own document collection.



\---



\## 2. Objectives



The main objectives of this project are:



\- Load and process real PDF documents.

\- Extract text from the documents.

\- Divide documents into manageable overlapping chunks.

\- Convert document chunks into numerical vector representations.

\- Store the vectors in ChromaDB.

\- Retrieve relevant document chunks for a user question.

\- Generate a grounded answer using the retrieved information.

\- Compare answers produced with RAG and without RAG.



\---



\## 3. Technologies Used



| Technology | Purpose |

|---|---|

| Python | Main programming language |

| PyPDF | Extract text from PDF documents |

| Scikit-learn | Generate TF-IDF vector representations |

| ChromaDB | Store and retrieve document vectors |

| NumPy | Numerical processing |

| Virtual Environment | Project dependency isolation |



\---



\## 4. Document Collection



The project uses three research papers related to Transformers, language models, and Retrieval-Augmented Generation.



\### Documents



1\. `attention\_transformer.pdf`

&#x20;  - Attention Is All You Need



2\. `language\_models.pdf`

&#x20;  - Language Models are Few-Shot Learners



3\. `rag\_survey.pdf`

&#x20;  - Retrieval-Augmented Generation for Large Language Models: A Survey



These documents provide a real document collection for testing the retrieval system.



\---



\## 5. RAG Workflow



The system follows the following workflow:



```text

PDF Documents

&#x20;     |

&#x20;     v

Text Extraction

&#x20;     |

&#x20;     v

Document Chunking

&#x20;     |

&#x20;     v

TF-IDF Vectorization

&#x20;     |

&#x20;     v

ChromaDB Vector Storage

&#x20;     |

&#x20;     v

User Question

&#x20;     |

&#x20;     v

Query Vectorization

&#x20;     |

&#x20;     v

Similarity Search

&#x20;     |

&#x20;     v

Relevant Document Chunks

&#x20;     |

&#x20;     v

Grounded Answer

