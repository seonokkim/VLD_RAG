# VLD-RAG: Visually-rich Long Document Retrieval-Augmented Generation

> **Work in Progress**  
> This project is currently under active development. Features and APIs may change.

Visually-rich documents such as reports, slides, and manuals often distribute the evidence needed to answer a question across multiple pages, mixing text with layout cues, tables, charts, and figures. This work studies multimodal retrieval-augmented generation for question answering over such visually-rich long documents, where retrieval must select evidence pages that include both textual and visual signals.

## Overview

VLD-RAG is an agentic multimodal RAG framework for multi-page evidence retrieval and cross-page reasoning over long documents. VLD-RAG builds a page-preserving multimodal index that stores parsed text, page-level metadata, and dense visual representations, and uses a hybrid retrieval strategy that combines keyword-based sparse search with dense semantic queries to identify candidate sources and evidence pages.

A verifier-guided agent workflow coordinates a Retrieval Agent, Answer Agent, and Validation Agent to broaden evidence coverage, detect missing citations, and refine retrieval requests when needed.

## Key Features

- **Page-preserving Multimodal Index**: Stores parsed text, page-level metadata, and dense visual representations
- **Hybrid Retrieval Strategy**: Combines keyword-based sparse search (BM25) with dense semantic queries
- **Agentic Workflow**: Coordinated Retrieval Agent, Answer Agent, and Validation Agent
- **Multi-page Evidence Retrieval**: Handles evidence scattered across multiple pages
- **Cross-page Reasoning**: Enables reasoning over evidence from multiple pages
- **Neon PostgreSQL Integration**: Database entities for document, page, chunk, and embedding management
- **Comprehensive Evaluation Metrics**: Retrieval evaluation metrics (Recall@K, MRR, nDCG)

## Project Structure

```
VLD_RAG/
├── parser/                    # Document parsing engines
│   ├── engines/               # Parser engine implementations
│   │   ├── paddle_ocr.py      # PaddleOCR parser with PP-StructureV3 support
│   │   └── __init__.py
│   ├── base.py                # Base parser interface
│   ├── schema.py              # Common schema definitions (PageParse, Block, BBox)
│   └── __init__.py
├── llm/                       # Vision-Language Model integrations
│   ├── base.py                # Base LLM interface
│   ├── qwen3_vl_4b_instruct.py  # Qwen3-VL-4B-Instruct wrapper
│   ├── internvl3_5_4b.py      # InternVL3-5-4B wrapper
│   └── __init__.py
├── retriever/                 # Retrieval components
│   ├── bm25_retriever.py      # BM25 sparse text retrieval
│   ├── colpali_vision_retriever.py  # ColPali vision-based retrieval
│   ├── vector_loader.py       # Vector embedding loader
│   ├── scorer.py              # Embedding similarity scorer
│   ├── db_context.py          # Database context for retrieval
│   └── __init__.py
├── database/                   # Database entities and utilities
│   ├── entities.py            # Neon PostgreSQL ORM entities (Peewee)
│   ├── vector_field.py        # Custom VectorField for pgvector support
│   └── __init__.py
├── eval/                       # Evaluation metrics
│   ├── retrieval_metrics.py   # Retrieval evaluation metrics (R@K, MRR, nDCG)
│   └── __init__.py
└── README.md
```

## Modules

### Parser Module (`parser/`)

Document parsing engines for extracting text, tables, figures, and layout information from visually-rich documents.

- **PaddleOCRParser**: PaddleOCR-based parser with PP-StructureV3 support
  - Chart recognition
  - Table/formula recognition
  - Document unwarping and orientation classification
  - Markdown/JSON export
  - Element-based normalization for RAG input

- **Schema**: Common output schema (`PageParse`, `Block`, `BBox`) for unified parser output

### LLM Module (`llm/`)

Vision-Language Model wrappers for question answering and image understanding.

- **Qwen3VL4BInstruct**: Qwen3-VL-4B-Instruct model wrapper
- **InternVL35_4B**: InternVL3-5-4B model wrapper
- **BaseLLM**: Base interface for all LLM implementations

### Retriever Module (`retriever/`)

Hybrid retrieval components combining sparse and dense search.

- **BM25Retriever**: BM25-based sparse text retrieval
- **ColPaliVisionRetriever**: ColPali vision-based dense retrieval
- **VectorLoader**: Vector embedding loader for semantic search
- **EmbeddingScorer**: Embedding similarity scoring
- **RetrieverDbContext**: Database context for retrieval operations

### Database Module (`database/`)

Neon PostgreSQL database entities using Peewee ORM.

- **TBDocument**: Document metadata and basic information
- **TBPage**: Page-level information within documents
- **TBChunk**: Chunk-level information (crops/regions from pages)
- **TBEmbedding**: Unified embedding table supporting:
  - Single-vector and multi-vector modes
  - Multiple vision encoders (SigLIP, ColPali, OmniEmbed, DSE, etc.)
  - pgvector support via `pooled_embedding_vector`
  - Qdrant collection tracking
  - Faiss index mapping
- **TBRun**: Run tracking for experiments
- **VectorField**: Custom field for pgvector support

### Evaluation Module (`eval/`)

Retrieval evaluation metrics for assessing retrieval performance.

- **Recall@K**: R@1, R@5, R@10
- **MRR@K**: Mean Reciprocal Rank at K (MRR@10, MRR@100)
- **nDCG@K**: Normalized Discounted Cumulative Gain (nDCG@5, nDCG@10)
- **Top-K Accuracy**: Retrieval-as-classification metrics
- **Batch Metrics**: Aggregate metrics across multiple queries

## Installation

Installation instructions will be added as the project progresses. See `setup/` directory for future installation guides.

### Dependencies

- Python 3.8+
- PaddleOCR (for document parsing)
- Transformers (for LLM models)
- Peewee ORM (for database)
- pgvector (for vector similarity search)
- rank-bm25 (for BM25 retrieval)
- NumPy, PIL (for image processing)

## Usage

Usage examples and documentation will be added as features are completed.

### Basic Example

```python
from parser.engines import PaddleOCRParser
from llm import Qwen3VL4BInstruct
from retriever import BM25Retriever

# Initialize parser
parser = PaddleOCRParser(device="cpu", use_chart_recognition=True)
parser.initialize()

# Parse document
page_parse = parser.parse_page(
    doc_id="doc_001",
    page_no=0,
    image=image,
    image_path="path/to/page.png"
)

# Initialize LLM
llm = Qwen3VL4BInstruct(model_path="../models/Qwen3-VL-4B-Instruct")

# Answer question
answer = llm.answer_question(image, "What is the main topic?")

# Initialize retriever
retriever = BM25Retriever(corpus=corpus)
results = retriever.retrieve(query="example query", top_k=5)
```

## Database Schema

The system uses Neon PostgreSQL with the following main tables:

- **tb_documents**: Document metadata
- **tb_pages**: Page-level information
- **tb_chunks**: Chunk-level information with status tracking
- **tb_embeddings**: Unified embedding storage (single-vector and multi-vector)
- **tb_runs**: Experiment run tracking

See `database/entities.py` for complete schema definitions.

## Evaluation

The evaluation module provides comprehensive retrieval metrics:

```python
from eval import recall_at_k, mrr_at_10, ndcg_at_10, calculate_all_metrics

# Calculate individual metrics
r_at_5 = recall_at_k(rankings, ground_truth, k=5)
mrr = mrr_at_10(rankings, ground_truth)
ndcg = ndcg_at_10(rankings, ground_truth)

# Calculate all metrics at once
metrics = calculate_all_metrics(
    rankings=rankings,
    ground_truth=ground_truth,
    k_values=[1, 5, 10],
    mrr_k_values=[10, 100],
    ndcg_k_values=[5, 10]
)
```

## Status

**This repository is a work-in-progress.** The project is being actively developed and refactored. Current implementation includes:

- ✅ Parser module with PaddleOCR integration
- ✅ LLM integration modules (Qwen3-VL, InternVL)
- ✅ Retriever components (BM25, ColPali, vector loader)
- ✅ Database integration (Neon PostgreSQL entities)
- ✅ Evaluation metrics (retrieval metrics)
- 🚧 Agent modules (in development)
- 🚧 Full pipeline integration (in development)

## Contributing

This project is in active development. Please note that APIs and structure may change.

## License

License information will be added.
