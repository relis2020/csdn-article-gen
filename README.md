Vector Store and Embedding System
Relevant source files
Purpose and Scope
The Vector Store and Embedding System provides semantic search capabilities for the article generation pipeline. It converts structured document blocks into dense vector embeddings and enables similarity-based retrieval. The system uses the Qwen3-Embed-0.6B model to generate 1024-dimensional embeddings and implements threshold-based similarity search rather than fixed top-k retrieval.

This document provides an overview of the vector storage architecture and embedding generation workflow. For detailed implementation of the VectorStore class and persistence mechanisms, see VectorStore Implementation. For embedding model integration details, see LocalEmbedder and Sentence Transformers. For model architecture and configuration, see Qwen3-Embed-0.6B Model Configuration.

Sources: 
csdn_article_generation/article_generator_system.py
15-165
 
csdn_article_generation/qwen3-embed-0.6b/README.md
1-50

System Architecture
The Vector Store and Embedding System operates as the knowledge retrieval layer between document processing and article generation. It consists of three primary components that work together to enable semantic search.

Component Architecture Diagram


















Sources: 
csdn_article_generation/article_generator_system.py
15-274

Key Components
LocalEmbedder Class
The LocalEmbedder class (
article_generator_system.py
15-33
) wraps the SentenceTransformer library to provide embedding generation using the Qwen3-Embed-0.6B model.

Attribute	Type	Description
model	SentenceTransformer	Instance of the Qwen3 embedding model
model_name	str	Default: "./qwen3-embed-0.6b"
Method	Parameters	Returns	Description
embed()	texts: List[str]	np.ndarray	Converts text list to embedding vectors
The embed() method automatically handles batching and normalization through the SentenceTransformer library, returning embeddings of shape (n_texts, 1024).

Sources: 
csdn_article_generation/article_generator_system.py
15-33

VectorStore Class
The VectorStore class (
article_generator_system.py
36-165
) manages embedding storage, index building, and similarity-based retrieval for a single document.

Attribute	Type	Description
embedder	LocalEmbedder	Instance used for generating embeddings
embeddings	np.ndarray	Matrix of embedding vectors (shape: n × 1024)
questions	List[str]	List of paranames from JSON blocks
answers	List[str]	List of content from JSON blocks
data	List[Dict]	Original JSON block data with metadata
index_dir	str	Directory path for persistent storage
Sources: 
csdn_article_generation/article_generator_system.py
36-45

MultiVectorStoreManager Class
The MultiVectorStoreManager class (
article_generator_system.py
239-274
) manages multiple VectorStore instances, one per document, enabling cross-document retrieval.

Attribute	Type	Description
json_files	List[str]	List of JSON file paths to load
embedder	LocalEmbedder	Shared embedder instance across all stores
stores	Dict[str, VectorStore]	Dictionary mapping document names to VectorStore instances
base_index_dir	str	Base directory for all index subdirectories
Sources: 
csdn_article_generation/article_generator_system.py
239-251

Data Flow Pipeline
The following diagram illustrates how document blocks flow through the embedding and storage pipeline:

Embedding and Storage Pipeline


















Sources: 
csdn_article_generation/article_generator_system.py
46-130

Embedding Generation Process
Model Architecture
The Qwen3-Embed-0.6B model is a specialized text embedding model with the following specifications:

Parameter	Value
Model Size	0.6 billion parameters
Number of Layers	28
Context Length	32,768 tokens
Embedding Dimension	1024 (configurable down to 32)
Pooling Strategy	Last-token pooling
Base Model	Qwen/Qwen3-0.6B-Base
Sources: 
csdn_article_generation/qwen3-embed-0.6b/README.md
30-44

Embedding Workflow
The embed() method in LocalEmbedder performs the following operations:

Embedding Generation Sequence

The model automatically handles:

Tokenization with left padding
Batch processing of multiple texts
L2 normalization of output vectors
Conversion to NumPy format
Sources: 
csdn_article_generation/article_generator_system.py
26-33
 
csdn_article_generation/qwen3-embed-0.6b/README.md
117-126

Vector Storage and Retrieval
Index Building
The build_index() method (
article_generator_system.py
107-130
) constructs the searchable index with batch processing to manage memory:

Cache Check: Attempts to load existing index via load_index() (
line 109
)
Batch Embedding: Processes texts in batches (default: 32) to reduce memory pressure (
lines 118-124
)
Vector Stacking: Combines batch embeddings using np.vstack() (
line 126
)
Persistence: Saves to disk via save_index() (
line 130
)
Sources: 
csdn_article_generation/article_generator_system.py
107-130

Persistence Mechanism
The VectorStore uses a dual-file persistence strategy:

File	Format	Contents	Generation Method
embeddings.npy	NumPy binary	Embedding matrix (n × 1024)	np.save() (
line 66
)
metadata.pkl	Python pickle	{questions, answers, data} dict	pickle.dump() (
line 76
)
This separation optimizes loading performance by allowing NumPy's efficient array loading for the large embedding matrix while keeping metadata in a flexible dictionary format.

Sources: 
csdn_article_generation/article_generator_system.py
59-78
 
csdn_article_generation/article_generator_system.py
80-102

Similarity Search
The search_with_threshold() method (
article_generator_system.py
132-165
) implements threshold-based retrieval using cosine similarity:

Search Algorithm Flow









Key characteristics:

Uses cosine similarity: dot(a, b) / (||a|| * ||b||)
Filters results by minimum threshold (default: 0.3)
Returns all results above threshold, up to max_results limit
Includes source metadata in each result for attribution
Sources: 
csdn_article_generation/article_generator_system.py
132-165

Multi-Store Architecture
Store Management
The MultiVectorStoreManager creates isolated VectorStore instances for each document, enabling independent index management:

Multi-Store Initialization Process













Sources: 
csdn_article_generation/article_generator_system.py
240-251

Cross-Document Search
The search_all_stores() method (
article_generator_system.py
264-274
) performs parallel search across all document stores:

Step	Operation	Code Reference
1	Initialize result list	all_results = [] (line 266)
2	Iterate over stores	for name, store in self.stores.items() (line 267)
3	Search each store	store.search_with_threshold(query, threshold, max_results) (line 269)
4	Add source attribution	result['source'] = name (line 272)
5	Aggregate results	all_results.extend(results) (line 273)
Each result includes a source field identifying which document it came from, enabling proper attribution in generated articles.

Sources: 
csdn_article_generation/article_generator_system.py
264-274

Index Directory Structure
The multi-store system creates the following directory hierarchy:

vector_indexes/                    # base_index_dir
├── doc1_name/                     # One directory per document
│   ├── embeddings.npy            # Embedding matrix
│   └── metadata.pkl              # Questions, answers, data
├── doc2_name/
│   ├── embeddings.npy
│   └── metadata.pkl
└── doc3_name/
    ├── embeddings.npy
    └── metadata.pkl
Each document's index is independently loadable and rebuildable without affecting other documents.

Sources: 
csdn_article_generation/article_generator_system.py
240-262

Integration with Article Generation
The vector store system integrates with the ArticleGenerator class through the following initialization and usage pattern:

Integration Architecture










The ArticleGenerator uses the vector store system to:

Execute semantic searches based on dynamically generated queries
Retrieve relevant content blocks with similarity scores
Track source attribution for proper citation
Accumulate context across multiple search iterations
Sources: 
csdn_article_generation/article_generator_system.py
277-688
